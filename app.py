import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import numpy as np
import streamlit as st
from simulation import VesselConfig,solve_hemodynamics,simulate_particles,perfusion_metrics

st.set_page_config(page_title="Blood Vessel Network",page_icon="🩸",layout="wide")
st.title("Blood Vessel Network Perfusion")
st.caption("Branching-network model with pressure, flow, velocity, shear stress, tracer transport, and occlusion.")

with st.sidebar:
    levels=st.slider("Branching levels",2,6,4)
    angle=st.slider("Branch angle (degrees)",12.0,45.0,28.0,1.0)
    root_radius=st.slider("Root radius (µm)",25.0,100.0,55.0,5.0)
    rr=st.slider("Radius ratio",0.55,0.95,0.78,0.01)
    lr=st.slider("Length ratio",0.50,0.90,0.74,0.01)
    viscosity=st.slider("Blood viscosity (mPa·s)",2.0,8.0,3.5,0.1)
    pin=st.slider("Inlet pressure (Pa)",2000.0,9000.0,4500.0,100.0)
    pout=st.slider("Outlet pressure (Pa)",500.0,3000.0,1200.0,100.0)
    max_seg=2**levels-1
    occ=st.selectbox("Occluded segment",[-1]+list(range(max_seg)),
        format_func=lambda x:"None" if x==-1 else f"Segment {x}")
    occf=st.slider("Radius reduction",0.0,0.95,0.0,0.05)
    particles=st.slider("Tracer particles",20,500,180,10)
    duration=st.slider("Animation duration (s)",1.0,20.0,8.0,0.5)
    run=st.button("Run simulation",type="primary",use_container_width=True)

config=VesselConfig(levels=levels,branch_angle_deg=angle,root_radius_um=root_radius,
 radius_ratio=rr,length_ratio=lr,viscosity_pa_s=viscosity*1e-3,
 inlet_pressure_pa=pin,outlet_pressure_pa=pout,occluded_segment=occ,
 occlusion_fraction=occf,particle_count=particles,duration_s=duration)

if run or "sol" not in st.session_state:
    with st.spinner("Solving vessel network..."):
        st.session_state.sol=solve_hemodynamics(config)
        st.session_state.parts=simulate_particles(config,st.session_state.sol)
        st.session_state.cfg=config

sol=st.session_state.sol; parts=st.session_state.parts; active=st.session_state.cfg
if active!=config: st.info("Parameters changed. Click Run simulation to update.")
metrics=perfusion_metrics(sol)
a,b,c,d=st.columns(4)
a.metric("Total inflow",f"{metrics['total_inflow_nL_s']:.2f} nL/s")
b.metric("Mean terminal flow",f"{metrics['mean_terminal_flow_nL_s']:.2f} nL/s")
c.metric("Perfusion heterogeneity",f"{metrics['terminal_flow_cv']:.3f} CV")
d.metric("Minimum terminal flow",f"{metrics['minimum_terminal_flow_nL_s']:.2f} nL/s")

segs=sol["segments"]; nodes=sol["nodes"]
tabs=st.tabs(["Flow map","Pressure & shear","Particle transport","Perfusion analysis","Export & assumptions"])

def network_plot(column,title,label):
    fig,ax=plt.subplots(figsize=(12,7))
    lines=[[(r.x0_um,r.y0_um),(r.x1_um,r.y1_um)] for r in segs.itertuples()]
    widths=1+5*segs["effective_radius_um"].to_numpy()/segs["effective_radius_um"].max()
    lc=LineCollection(lines,array=segs[column].to_numpy(),linewidths=widths)
    ax.add_collection(lc); ax.scatter(nodes["x_um"],nodes["y_um"],s=10)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.colorbar(lc,ax=ax,label=label)
    return fig

with tabs[0]:
    fig=network_plot("flow_nL_s","Branch flow distribution","Flow (nL/s)")
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    fig=network_plot("mean_velocity_mm_s","Mean blood velocity","Velocity (mm/s)")
    st.pyplot(fig,use_container_width=True); plt.close(fig)

with tabs[1]:
    fig=network_plot("pressure_start_pa","Pressure distribution","Pressure (Pa)")
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    fig=network_plot("wall_shear_pa","Wall shear stress","Shear stress (Pa)")
    st.pyplot(fig,use_container_width=True); plt.close(fig)

with tabs[2]:
    if parts.empty:
        st.warning("No tracer paths generated.")
    else:
        t=st.slider("Tracer time (s)",0.0,float(active.duration_s),float(active.duration_s),0.05)
        tol=max(active.dt_s*1.5,0.04)
        visible=parts[np.abs(parts["time_s"]-t)<=tol]
        fig,ax=plt.subplots(figsize=(12,7))
        for r in segs.itertuples():
            ax.plot([r.x0_um,r.x1_um],[r.y0_um,r.y1_um],linewidth=max(1,r.effective_radius_um/12),alpha=.45)
        if not visible.empty: ax.scatter(visible["x_um"],visible["y_um"],s=24)
        ax.autoscale(); ax.set_aspect("equal"); ax.set_title(f"Tracer particles at t={t:.2f} s")
        st.pyplot(fig,use_container_width=True); plt.close(fig)

        if st.button("Generate blood-flow GIF"):
            fig,ax=plt.subplots(figsize=(10,6))
            for r in segs.itertuples():
                ax.plot([r.x0_um,r.x1_um],[r.y0_um,r.y1_um],linewidth=max(1,r.effective_radius_um/12),alpha=.4)
            ax.autoscale(); ax.set_aspect("equal"); sc=ax.scatter([],[],s=22)
            times=np.linspace(0,active.duration_s,120)
            def update(tt):
                cur=parts[np.abs(parts["time_s"]-tt)<=tol]
                sc.set_offsets(cur[["x_um","y_um"]].to_numpy() if not cur.empty else np.empty((0,2)))
                ax.set_title(f"Blood-flow tracer transport: t={tt:.2f} s")
                return [sc]
            ani=FuncAnimation(fig,update,frames=times,interval=70)
            path="blood_vessel_network.gif"; ani.save(path,writer=PillowWriter(fps=12))
            plt.close(fig); data=open(path,"rb").read()
            st.image(data); st.download_button("Download GIF",data=data,file_name=path,mime="image/gif")

with tabs[3]:
    terminal=sol["terminal_segments"].copy()
    terminal["outlet"]=terminal["end_node"].astype(str)
    st.bar_chart(terminal.set_index("outlet")[["flow_nL_s"]])
    st.dataframe(segs[["segment_id","level","effective_radius_um","flow_nL_s",
        "mean_velocity_mm_s","wall_shear_pa","pressure_start_pa","pressure_end_pa"]],use_container_width=True)

with tabs[4]:
    st.download_button("Download segment results",segs.to_csv(index=False).encode(),"blood_vessel_segments.csv","text/csv")
    st.download_button("Download node pressures",nodes.to_csv(index=False).encode(),"blood_vessel_nodes.csv","text/csv")
    st.download_button("Download tracer trajectories",parts.to_csv(index=False).encode(),"blood_vessel_particles.csv","text/csv")
    st.markdown('''### Assumptions
- Rigid cylindrical vessels
- Steady laminar Newtonian flow
- Hagen-Poiseuille resistance
- Flow conservation at junctions
- Particle routing follows local flow fractions
- Occlusion reduces one segment radius
- No pulsatility, compliance, red-cell effects, or autoregulation

This is a reduced-order portfolio model, not patient-specific CFD.''')
