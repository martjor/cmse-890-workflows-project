import matplotlib.pyplot as plt 
import numpy as np

trajectories = np.load(snakemake.input[0])

# Take average over all trajectories
trajectories = np.mean(trajectories,axis=0)

# Plot each trajectory
colors = {
    0: "blue",
    1: "red",
    2: "green"
}

fig, ax = plt.subplots(figsize=(6,3),dpi=300)
for idx, color in colors.items():
    ax.plot(trajectories[idx,:],color=colors[idx])
    
ax.legend(["S","I","R"])
ax.set_ylabel("Population fraction")
ax.set_xlabel("t")
plt.savefig(snakemake.output[0])


