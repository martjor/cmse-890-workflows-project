from scipy.sparse import load_npz, save_npz
import networkx as nx
from yaml import safe_load, dump
import logging

def save_graph(file,G):
    '''Saves the adjacency matrix of a graph'''
    save_npz(file,nx.to_scipy_sparse_array(G))

def load_graph(file):
    '''Loads the adjacency matrix of a graph'''
    return nx.from_scipy_sparse_array(load_npz(file))

def save_dict(file,dictionary):
    '''Saves a dictionary in yaml format
    '''
    with open(file,'w') as file: 
        dump(dictionary,file,default_flow_style=False)
        
def load_dict(file):
    '''Loads a dictionary from a file
    '''
    with open(file,'r') as file:
        return safe_load(file)
    
# Redirect stdout and stderr to logging
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        if message.strip():  # Ignore empty lines
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass  # Required for compatibility with sys.stdout/sys.stderr
    
    