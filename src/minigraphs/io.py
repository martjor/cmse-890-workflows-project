import numpy as np
import pandas as pd 
import os 
from glob import glob
import json 

class Reader():
    def __init__(self, DATA_DIR):
        self.root = DATA_DIR
        
    @property
    def graph(self):
        return self._graph
    
    @graph.setter
    def graph(self, graph_name) -> None:
        graph_path = os.path.join(self.root,'networks',graph_name)
        if not os.path.exists(graph_path):
            raise Exception(f"Graph specified at '{graph_path}' does not exist")
            
        self._graph = graph_name
        self._NET_DIR = os.path.join(graph_path)
        self._MIN_DIR = os.path.join(self._NET_DIR,'miniatures')
        
        print(f"Pointing to {graph_name} at {graph_path}\n")
    
    @property
    def metrics(self) -> str:
        metrics_file = os.path.join(self._NET_DIR,'metrics.json')
        if not os.path.exists(metrics_file):
            raise Error("Metrics file not found")
        
        with open(metrics_file,'r') as file:
            metrics = json.load(metrics_file)
            
        return metrics
    
    def list_miniatures(self,pattern='*'):
        miniatures = glob(pattern,root_dir=self._MIN_DIR)
        miniatures.sort()
        
        return miniatures
    
    @property
    def miniature(self):
        # Path to miniature metrics
        metrics_path = os.path.join(self._min_path,'metrics.json')
        
        with open(metrics_path,'r') as file:
            metrics = json.load(file)
        
        # Paths to trajectories
        dfs = [pd.read_parquet(file_name) for file_name in glob(os.path.join(self._min_path,"*.parquet"))]
        
        dfs = pd.concat(dfs,keys=np.arange(len(dfs)))
        dfs.reset_index(inplace=True,names=['Replica','Step'])
        
        # Calculate errors
        for key in ['density','assortativity_norm','clustering']:
            dfs['err_' + key] = np.abs(dfs[key] - metrics[key])/metrics[key] * 100
        
        return metrics, dfs
    
    @miniature.setter
    def miniature(self,miniature_name):
        if not (miniature_name in self.list_miniatures()):
            raise Exception("Miniature not found in miniatures directory")
        
        self._miniature = miniature_name
        self._min_path = os.path.join(self._MIN_DIR,miniature_name)
        
        print(f"Pointing to miniature {miniature_name} at {self._min_path}")