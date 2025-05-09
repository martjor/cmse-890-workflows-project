import networkx as nx 
from minigraphs.graph import spectral_radius, degree_distribution_moment

DICT_METRICS_FUNCS = {
    'n_nodes': lambda graph : graph.number_of_nodes(),
    'density': nx.density,
    'sparsity': lambda graph: graph.number_of_nodes() / graph.number_of_edges(),
    'assortativity_norm': lambda graph: (nx.degree_assortativity_coefficient(graph)+1)/2,
    'clustering': nx.average_clustering,
    'eig_1': lambda graph: spectral_radius(graph),
    'dd_ratio': lambda graph: degree_distribution_moment(graph,2) / degree_distribution_moment(graph)
}