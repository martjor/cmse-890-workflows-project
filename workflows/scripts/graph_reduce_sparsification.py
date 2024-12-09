import networkx as nx
from utils import load_graph, save_graph
import yaml

# INPUTS
adjacency_file = snakemake.input[0]
targets_file = snakemake.input[1]
reduction_file = snakemake.output[0]

with open(targets_file,'r') as file:
    params = yaml.safe_load(file)

# Load Graph
graph = load_graph(adjacency_file)

# Sparsify graph
graph_sparse = nx.spanner(graph, params['stretch'])

# Store graph
save_graph(reduction_file, graph_sparse)