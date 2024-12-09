from minigraphs.miniaturize import NX_ASSORTATIVITY, NX_CLUSTERING, NX_DENSITY
import networkx as nx 
from yaml import dump
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigs
from numpy import real

def graph_components(G):
    # Get connected components of the graph
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    
    # Generate induced graphs
    graphs = [G.subgraph(c).copy() for c in components]

    return graphs

# PREAMBLE 
adjacency_file = snakemake.input[0]
metrics_file = snakemake.output[0]

functions = {
    'n_nodes': lambda G: G.number_of_nodes(),
    'n_edges': lambda G: G.number_of_edges(),
    'density': NX_DENSITY,
    'clustering': NX_CLUSTERING,
    'assortativity': NX_ASSORTATIVITY,
}

# LOAD ADJACENCY MATRIX AND GENERATE GRAPH
adjacency = load_npz(adjacency_file)
adjacency = adjacency._asfptype()
graph = nx.from_scipy_sparse_array(adjacency)

# CALCULATE METRICS
components = graph_components(graph)

evals,_ = eigs(adjacency,2)
print(evals)

metrics = {metric: func(graph) for metric, func in functions.items()}
metrics['n_components'] = len(components)
metrics['connectivity'] = components[0].number_of_nodes() / graph.number_of_nodes()
metrics['eig_1'] = float(real(evals[0]))
metrics['eig_2'] = float(real(evals[1]))

# WRITE METRICS
with open(metrics_file,'w') as file:
    dump(metrics,file,default_flow_style=False)
    