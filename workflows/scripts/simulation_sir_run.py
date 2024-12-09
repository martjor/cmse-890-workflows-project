from minigraphs import simulation as sim
from scipy.sparse import load_npz
import numpy as np
import yaml

# Inputs
adjacency_file = snakemake.input[0]
#parameters_file = snakemake.input[1]
trajectories_file = snakemake.output[0]
parameters = snakemake.params.config
# with open(parameters_file,'r') as file:
#     parameters = yaml.safe_load(file)
    
# Preamble
tau = parameters['tau']
gamma = parameters['gamma']
n_steps = parameters['n_steps']
n_trials = parameters['n_trials']

sir = sim.Sir(tau, gamma)

# Instantiate simulation object
simulation = sim.Simulation(load_npz(adjacency_file))

# Allocate memory
shape = (n_trials, 3, n_steps)
results = np.zeros(shape)

# Simulate epidemic
for i in range(n_trials):
    simulation.run(sir, n_steps)
    
    results[i,:,:] = simulation.trajectories_.T
    
# Save simulation results
np.save(trajectories_file,results)