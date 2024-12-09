import numpy as np 
import matplotlib.pyplot as plt 

# PREAMBLE
dist_file = snakemake.input[0]
plot_file = snakemake.output[0]
quantity = snakemake.wildcards.quantity

# LOAD DISTRIBUTION
distribution = np.load(dist_file)
mean = (distribution[:,0] * distribution [:,1]).sum()

# PLOT DISTRIBUTION
fig, ax = plt.subplots(figsize=(3,3),dpi=300)
ax.bar(distribution[:,0],distribution[:,1])
ax.set_title(f"{quantity.capitalize()} distribution")
ax.set_xlabel(f"{quantity.capitalize()}")
ax.axvline(mean,linestyle='--',color='r',label="Mean")

plt.savefig(plot_file,bbox_inches='tight')