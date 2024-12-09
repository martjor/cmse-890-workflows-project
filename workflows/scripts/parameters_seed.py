import numpy as np
import pandas as pd

def cartesian_product(*arrays):
    meshgrids = np.meshgrid(*arrays,indexing='ij')
    cart_prod = np.stack(meshgrids,axis=-1)
    cart_prod = cart_prod.reshape(-1,len(arrays))
    return cart_prod

# Preamble
config = snakemake.params.config

# Calculate cartesian product of input parameters
parameters = cartesian_product(*[eval(param) for param in config.values()])

# Create DataFrame
parameters = pd.DataFrame(parameters,columns=config.keys())
parameters.to_csv(output[0],index=False)