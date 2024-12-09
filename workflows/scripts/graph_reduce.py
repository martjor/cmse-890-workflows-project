import minigraphs as mini
import networkx as nx 
from yaml import dump
from utils import load_graph, save_graph, save_dict

def coarsening(graph,alpha):
    '''Coarsens the graph
    '''
    coarsener = mini.CoarseNET(alpha,graph)
    coarsener.coarsen()
    
    return coarsener.G_coarse_

def sparsification(graph,stretch):
    '''Sparsifies the graph
    '''
    return nx.spanner(graph,stretch)

methods = {
    'coarsening': coarsening,
    'sparsification': sparsification
}

# INPUTS AND PARAMETERS
adjacency_file = snakemake.input[0]
adjacency_reduced_file = snakemake.output[0]
parameters_file = snakemake.output[1]

method = snakemake.wildcards.method 
method_parameters = snakemake.params.method_parameters

# REDUCE
graph = load_graph(adjacency_file)

parameters = list(method_parameters.values())
graph_reduced = methods[method](graph,*parameters)

# SAVE GRAPH & METRICS
save_graph(adjacency_reduced_file, graph_reduced)
save_dict(parameters_file, method_parameters)
