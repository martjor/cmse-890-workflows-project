import networkx as nx 
from minigraphs.graph import spectral_radius, degree_distribution_moment
from minigraphs import changes 

DICT_METRICS_FUNCS = {
    'n_nodes': nx.number_of_nodes,
    'n_edges': nx.number_of_edges,
    'density': nx.density,
    'assortativity': nx.degree_assortativity_coefficient,
    'assortativity_norm': lambda graph: (nx.degree_assortativity_coefficient(graph) + 1)/2,
    'clustering': nx.average_clustering,
    'eig_1': lambda graph: spectral_radius(graph),
    'dd_ratio': lambda graph: degree_distribution_moment(graph,2) / degree_distribution_moment(graph)
}

dict_changes = {
    'add': changes.Add,
    'remove': changes.Remove,
    'switch': changes.Switch,
    'random': changes.ChangeSampler(
        changes=[
            ('add', changes.Add, 1/4),
            ('remove', changes.Remove, 1/4),
            ('switch', changes.Switch, 1/2)
        ]
    )
}