from minigraphs.miniaturize import CoarseNET
import yaml
from utils import load_graph, save_graph

# Preamble
adjacency_file = snakemake.input[0]
targets_file = snakemake.input[1]
reduction_file = snakemake.output[0]

with open(targets_file,'r') as file:
    targets = yaml.safe_load(file)
    
# Load graph
graph = load_graph(adjacency_file)

# Instantiate CoarseNET
coarsener = CoarseNET(targets['alpha'],graph)
coarsener.coarsen()

# Save graph
save_graph(reduction_file, coarsener.G_coarse_)