from utils import load_dict
import pandas as pd

# INPUTS
files = snakemake.input.files
df_file = snakemake.output[0]

# AGGREGATE DICTIONARIES

if "qois" in files[0]:
    dfs = [pd.DataFrame(load_dict(file)).T.stack() for file in files]
    df =pd.concat(dfs,axis=1).T 
else:
    dfs = [load_dict(file) for file in files]
    df = pd.DataFrame(dfs)
    
df.rename(lambda idx: f"graph_{idx}",inplace=True)

# SAVE DF
df.to_csv(df_file)