import numpy as np 
from scripts.utils.io import save_dict

infected = np.load(snakemake.input[0])[:,1,:].mean(axis=0)

# Obtain peak of each epidemic
peak = infected.max()

if peak > 0.1:
    results = {
        'peak': float(peak),
        'time': float(infected.argmax())
    }

else:
    results = {
        'peak': -1,
        'time': -1,
    }

save_dict(snakemake.output[0], results)







