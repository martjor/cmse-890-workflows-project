import networkx as nx 
from numpy import inf, load
from minigraphs.miniaturize import MH, NX_ASSORTATIVITY
from utils import save_graph
from yaml import dump

def assortativity_graph(n_vertices,assortativity):
    '''Create a graph with the specified assortativity
    '''
    # Preamble
    beta = inf 
    epsilon = 0.01
    p = 0.04

    funcs = {
        'assortativity': NX_ASSORTATIVITY
    }

    targets = {
        'assortativity': assortativity
    }

    # Instantiate annealer
    annealer = MH(funcs)

    # Instantiate ER grpah
    G = nx.erdos_renyi_graph(n_vertices,p)

    # Optimize graph
    annealer.transform(G,targets,beta=beta,epsilon=epsilon)

    return annealer.graph_

# ------------------- PREAMBLE
n_vertices = snakemake.params.n_vertices
index = snakemake.wildcards.index
metric = snakemake.wildcards.metric
adjacency_file = snakemake.output.adjacency_file
target_file = snakemake.output.target_file
input_file = snakemake.input[0]

# Instantiate generators
generator = {
    'density': lambda p: nx.erdos_renyi_graph(n_vertices, p),
    'clustering': lambda p: nx.erdos_renyi_graph(n_vertices,p),
    'assortativity': lambda p: assortativity_graph(n_vertices,p)
}

# -------------------- LOAD PARAMETERS
parameters = load(input_file,mmap_mode='r')

# -------------------- GENERATE GRAPH
p = float(parameters[int(index)])
G = generator[metric](p)

# -------------------- SAVE GRAPH & SEED
save_graph(adjacency_file,G)
with open(target_file,'w') as file:
    dump({'target': p}, file, default_flow_style=False)
