from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class VesselConfig:
    levels: int = 4
    root_length_um: float = 700.0
    root_radius_um: float = 55.0
    length_ratio: float = 0.74
    radius_ratio: float = 0.78
    branch_angle_deg: float = 28.0
    viscosity_pa_s: float = 0.0035
    inlet_pressure_pa: float = 4500.0
    outlet_pressure_pa: float = 1200.0
    occluded_segment: int = -1
    occlusion_fraction: float = 0.0
    particle_count: int = 180
    duration_s: float = 8.0
    dt_s: float = 0.02
    seed: int = 21

def build_network(c):
    nodes=[{"node_id":0,"x_um":0.0,"y_um":0.0,"level":0}]
    segments=[]
    frontier=[(0,0.0)]
    next_node=1
    sid=0
    for level in range(c.levels):
        new=[]
        L=c.root_length_um*(c.length_ratio**level)
        r=c.root_radius_um*(c.radius_ratio**level)
        for parent,angle in frontier:
            parent_node=nodes[parent]
            angles=[angle] if level==0 else [angle+math.radians(c.branch_angle_deg), angle-math.radians(c.branch_angle_deg)]
            for a in angles:
                x1=parent_node["x_um"]+L*math.cos(a)
                y1=parent_node["y_um"]+L*math.sin(a)
                nodes.append({"node_id":next_node,"x_um":x1,"y_um":y1,"level":level+1})
                er=r*(1-c.occlusion_fraction) if sid==c.occluded_segment else r
                er=max(er,0.02*r)
                segments.append({"segment_id":sid,"start_node":parent,"end_node":next_node,
                    "level":level,"x0_um":parent_node["x_um"],"y0_um":parent_node["y_um"],
                    "x1_um":x1,"y1_um":y1,"length_um":L,"radius_um":r,
                    "effective_radius_um":er,"angle_rad":a})
                new.append((next_node,a))
                next_node+=1; sid+=1
        frontier=new
    return pd.DataFrame(nodes),pd.DataFrame(segments)

def solve_hemodynamics(c):
    nodes,segs=build_network(c)
    terminal=set(nodes.loc[~nodes["node_id"].isin(segs["start_node"]),"node_id"])
    R=[]
    for row in segs.itertuples():
        L=row.length_um*1e-6
        r=row.effective_radius_um*1e-6
        R.append(8*c.viscosity_pa_s*L/(math.pi*r**4))
    segs["resistance_pa_s_m3"]=R
    segs["conductance_m3_pa_s"]=1/segs["resistance_pa_s_m3"]

    unknown=[int(n) for n in nodes["node_id"] if n!=0 and n not in terminal]
    idx={n:i for i,n in enumerate(unknown)}
    A=np.zeros((len(unknown),len(unknown))); b=np.zeros(len(unknown))
    adj={int(n):[] for n in nodes["node_id"]}
    for row in segs.itertuples():
        g=row.conductance_m3_pa_s
        adj[row.start_node].append((row.end_node,g))
        adj[row.end_node].append((row.start_node,g))
    for n in unknown:
        i=idx[n]
        for nb,g in adj[n]:
            A[i,i]+=g
            if nb in idx: A[i,idx[nb]]-=g
            elif nb==0: b[i]+=g*c.inlet_pressure_pa
            else: b[i]+=g*c.outlet_pressure_pa
    p=np.full(len(nodes),c.outlet_pressure_pa)
    p[0]=c.inlet_pressure_pa
    if unknown:
        sol=np.linalg.solve(A,b)
        for n,val in zip(unknown,sol): p[n]=val

    q=[]; vel=[]; shear=[]
    for row in segs.itertuples():
        flow=(p[row.start_node]-p[row.end_node])/row.resistance_pa_s_m3
        rm=row.effective_radius_um*1e-6
        q.append(flow)
        vel.append(flow/(math.pi*rm**2))
        shear.append(4*c.viscosity_pa_s*abs(flow)/(math.pi*rm**3))
    segs["pressure_start_pa"]=[p[int(n)] for n in segs["start_node"]]
    segs["pressure_end_pa"]=[p[int(n)] for n in segs["end_node"]]
    segs["flow_m3_s"]=q
    segs["flow_nL_s"]=np.array(q)*1e12
    segs["mean_velocity_m_s"]=vel
    segs["mean_velocity_mm_s"]=np.array(vel)*1e3
    segs["wall_shear_pa"]=shear
    nodes["pressure_pa"]=p
    terminal_segments=segs[segs["end_node"].isin(list(terminal))].copy()
    return {"nodes":nodes,"segments":segs,"terminal_segments":terminal_segments,
            "total_inflow_m3_s":float(segs.loc[segs["start_node"]==0,"flow_m3_s"].sum())}

def simulate_particles(c,solution):
    rng=np.random.default_rng(c.seed)
    segs=solution["segments"]
    by_start={int(n):g.copy() for n,g in segs.groupby("start_node")}
    rows=[]
    for pid in range(c.particle_count):
        node=0; time=0.0
        while node in by_start and time<=c.duration_s:
            options=by_start[node]
            w=np.abs(options["flow_m3_s"].to_numpy()); w=w/w.sum()
            seg=options.iloc[rng.choice(len(options),p=w)]
            travel=(seg["length_um"]*1e-6)/max(abs(seg["mean_velocity_m_s"]),1e-9)
            samples=max(2,int(math.ceil(travel/c.dt_s)))
            for s in range(samples):
                frac=s/(samples-1)
                t=time+frac*travel
                if t>c.duration_s: break
                rows.append({"particle_id":pid,"time_s":t,"segment_id":int(seg["segment_id"]),
                    "x_um":seg["x0_um"]+frac*(seg["x1_um"]-seg["x0_um"]),
                    "y_um":seg["y0_um"]+frac*(seg["y1_um"]-seg["y0_um"]),
                    "velocity_mm_s":seg["mean_velocity_mm_s"]})
            time+=travel; node=int(seg["end_node"])
    return pd.DataFrame(rows)

def perfusion_metrics(solution):
    f=np.abs(solution["terminal_segments"]["flow_nL_s"].to_numpy())
    mean=float(f.mean()) if len(f) else 0.0
    return {"total_inflow_nL_s":solution["total_inflow_m3_s"]*1e12,
            "mean_terminal_flow_nL_s":mean,
            "terminal_flow_cv":float(f.std()/mean) if mean>0 else 0.0,
            "minimum_terminal_flow_nL_s":float(f.min()) if len(f) else 0.0,
            "maximum_terminal_flow_nL_s":float(f.max()) if len(f) else 0.0}
