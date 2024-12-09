import networkx as nx 
from utils import load_graph

graph = load_graph(snakemake.input[0])
nx.write_gexf(graph,snakemake.output[0])
