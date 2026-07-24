from simulation import VesselConfig,solve_hemodynamics,simulate_particles,perfusion_metrics
config=VesselConfig(levels=4,occluded_segment=5,occlusion_fraction=0.55)
solution=solve_hemodynamics(config)
particles=simulate_particles(config,solution)
print(perfusion_metrics(solution))
solution["segments"].to_csv("vessel_segments.csv",index=False)
particles.to_csv("vessel_particles.csv",index=False)
