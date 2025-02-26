from minigraphs.reduction import MH 
from minigraphs.changes import Switch
from minigraphs.callback import EarlyStoppingCallback
from scripts.utils.io import save_graph
import networkx as nx

# Construct an annealer that only switches the edges:
annealer = MH(
    metrics_functions={'clustering': nx.average_clustering},
    max_iterations=snakemake.params.n_iterations,
    change=Switch,
    callbacks=[EarlyStoppingCallback(tol=0.01)]
)

# Construct graph
graph = nx.erdos_renyi_graph(
    snakemake.params.n_vertices,
    snakemake.params.density,
)

# Optimize graph
annealer.spec_optimize(
    {'clustering': snakemake.params.clustering},
    graph
)

# Save graph
save_graph(snakemake.output[0], graph)

