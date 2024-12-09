import numpy as np
import pandas as pd
from yaml import dump

def moment(dist,p):
    '''Calculates the pth momenth of a distribution
    '''
    moment = ((dist[:,0] ** p) * dist[:,1]).sum()
    return moment

def moment_central(dist,mean,p):
    '''Calculates the pth central moment of a distribution
    '''
    moment = np.power(np.abs(dist[:,1] * (dist[:,0] - mean)),p).sum()
    return moment 

# PREAMBLE
dist_file = snakemake.input[0]
metrics_file = snakemake.output[0]

# Load distribution
dist = np.load(dist_file)

# Calculate statistics
mean = moment(dist,1)
var = moment_central(dist,mean,2)
std = np.sqrt(var)

stats = {
    'mean': mean,
    'std': std,
    'skewness': moment_central(dist,mean,3)/(std**3),
    'kurtosis': moment_central(dist,mean,4)/(std**4),
    'm_2': moment(dist,2),
    'm_3': moment(dist,3),
    'm_4': moment(dist,4),
}

# SAVE METRICS
stats = {key: float(value) for key, value in stats.items()}

with open(metrics_file,'w') as file:
    dump(stats,file,default_flow_style=False)