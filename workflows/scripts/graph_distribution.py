import networkx as nx 
import numpy as np
from utils import load_graph

def degree_distribution(graph):
    '''Computes the degree distribution of a graph
    '''
    hist = nx.degree_histogram(graph)
    dist = np.zeros((len(hist),2))
    dist[:,0] = np.arange(len(hist))
    dist[:,1] = hist
    
    # Normalize distribution
    dist[:,1] /= dist[:,1].sum()
    
    return dist 

def distance_distribution(graph):
    '''Computes the distribution of pairwise distances of a graph
    '''
    paths = dict(nx.all_pairs_shortest_path(graph))
    dist = {}
    
    # Obtain histogram
    for node_a in range(graph.number_of_nodes()):
        # Iterate through neighbors
        for path in paths[node_a].values():
            # Calculate path length
            length = len(path) - 1
            
            # Avoid self-loops
            if length != 0: 
                # Add length to dictionary
                if not length in dist:
                    dist[length] = 1
                else:
                    dist[length] += 1
                    
    # Obtain largest distance
    girth = max(dist.keys())
    
    # Allocate memory for the distribution
    dist_arr = np.zeros((girth,2))
    dist_arr[:,0] = np.arange(1,girth+1)
    dist_arr[:,1] = np.array(list(dist.values()))
        
    # Normalize distribution
    dist_arr[:,1] /= dist_arr[:,1].sum() * 2
    
    return dist_arr

# PREAMBLE
adjacency_file = snakemake.input[0]
dist_file = snakemake.output[0]
quantity = snakemake.wildcards.quantity

functions = {
    'degree': lambda G: degree_distribution(G),
    'distance': lambda G: distance_distribution(G)
}

# LOAD GRAPH
graph = load_graph(adjacency_file)

# CALCULATE DISTRIBUTION
distribution  = functions[quantity](graph)

# SAVE DISTRIBUTION
np.save(dist_file, distribution)
