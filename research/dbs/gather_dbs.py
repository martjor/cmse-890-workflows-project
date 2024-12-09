import os
import sys
import glob
import pandas as pd

# Reader class to open directories
class Reader:
    def __init__(self,dir) -> None:
        self.root_dir = dir
        
    def load_dbs(self, dbs_names):
        return {name : pd.read_csv(os.path.join(self.root_dir,name,"df.csv"),delimiter=",") for name in dbs_names}
    
    def report(self):
        print(os.listdir(self.root_dir))
        
        
DATA_DIR = "/Users/jorgeaugustomartinezortiz/Repos/paper/dev_metropolis/data"
root = os.path.join(DATA_DIR,'databases')
file_name = sys.argv[1]
names_dbs = sys.argv[2:]

# Read data frames
reader = Reader(root)
dicts = reader.load_dbs(names_dbs)

# Merge dictionaries into a single DF
df = pd.concat(dicts,names=['Type','Row'])
df.reset_index(inplace=True)
df.drop(columns=['Row'],inplace=True)

# Save dataframe
df.to_csv(file_name,',',index=False)
