import numpy as np
import pandas as pd

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
dist_files = snakemake.input.distributions
table_file = snakemake.output[0]
n_distributions = len(dist_files)

distributions = [{}] * n_distributions

for i, file in enumerate(dist_files):
    # Load distribution
    dist = np.load(file)
    
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
        'm_4': moment(dist,4)
    }
    
    distributions[i] = stats 
    
# CONSTRUCT DATAFRAME
distributions = pd.DataFrame(distributions)
distributions.rename(lambda idx: f"graph_{idx}",inplace=True)

# SAVE DATAFRAME
distributions.to_csv(table_file)

    
    


