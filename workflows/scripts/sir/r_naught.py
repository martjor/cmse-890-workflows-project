import numpy as np 
from scripts.utils.io import save_dict

infected = np.load(snakemake.input[0])[:,1,:]

# Obtain peak of each epidemic
peaks = infected.max(1)

# Identify simulations where peak ocurred
idx = peaks > 0.1

# Measure time to the peak for simulations where peak ocurred
time = infected[idx].argmax(1)

# Calculate number of new cases at each time
new_infected = np.diff(infected[idx], axis=1)

# Calculate R0 as the ratio of new infected to current infected
r0 = new_infected / infected[idx,:-1]

results = {
    'mean': float(r0.mean()),
    'std': float(r0.std())
}

save_dict(snakemake.output[0], results)







