from utils import load_dict
import pandas as pd

# INPUTS
files = snakemake.input.files
summary_file = snakemake.output[0]

# SUMMARIZE QOIs
dfs = [pd.DataFrame(load_dict(file)).T.stack() for file in files]
df = pd.concat(dfs,axis=1).T
df.rename(lambda idx: f"graph_{idx}",inplace=True)

# SAVE DF
df.to_csv(summary_file)