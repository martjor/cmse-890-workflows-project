import numpy as np
from utils import save_dict

# Load simulation runs
results = np.load(snakemake.input[0])

qois_arrays = {
    'time-to-peak': results[:,1,:].argmax(1,keepdims=True),
    'peak': results[:,1,:].max(1,keepdims=True),
    'size': results[:,2,-1]
}

# Calculate QOIs
qois = {}
for quantity, array in qois_arrays.items():
    temp = {
        'mean': array.mean(0),
        'std': array.std(0),
        'min': array.min(0),
        'max': array.max(0),
    }
    
    # Convert values to floats
    temp = {key: float(metric) for key, metric in temp.items()}
    
    qois[quantity] = temp
    
# Save quantities of interest
save_dict(snakemake.output[0],qois)
    