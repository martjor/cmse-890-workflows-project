from minigraphs.reduction import MH
from minigraphs.callback import LoggingCallback, EarlyStoppingCallback
import networkx as nx

# Graphs
graph = nx.erdos_renyi_graph(100, 0.1, seed=42)

functions = {
    'density': nx.density,
    'clustering': nx.average_clustering,
    'assortativity': nx.degree_assortativity_coefficient
}

def test_instantiation():
    annealer = MH(
        functions,
    )

    keys = annealer.metrics_functions.keys()
    assert 'density' in keys
    assert 'clustering' in keys
    assert 'assortativity' in keys

def test_compute_metrics():
    

    annealer = MH(
        functions,
    )

    metrics = annealer.compute_graph_metrics(graph)

    assert metrics['density'] == nx.density(graph)
    assert metrics['clustering'] == nx.average_clustering(graph)
    assert metrics['assortativity'] == nx.degree_assortativity_coefficient(graph)

def test_no_start():
    annealer = MH(
        functions,
        max_iterations=0
    )

    annealer.spec_optimize(
        annealer.compute_graph_metrics(graph),
        graph
    )

def test_iter_stop():
    n_iterations = 100
    annealer = MH(
        {'density': nx.density},
        max_iterations=n_iterations
    )

    annealer.spec_optimize(
        {'density': 0.20},
        graph
    )

    assert annealer._state.iteration == n_iterations
    assert annealer._state.iteration > 0

def test_tol_stop():
    annealer = MH(
        {'density': nx.density},
        max_iterations=10000,
        callbacks=[
            LoggingCallback(), 
            EarlyStoppingCallback()
        ]
    )

    annealer.spec_optimize(
        {'density': 0.5},
        graph
    )

    print(nx.density(annealer.miniature__))
    assert annealer._state.iteration > 0
    assert annealer.distance__ < 0.05

    loss = annealer.log__['loss'].diff().dropna()
    assert (loss <= 0).all()

def test_warm_start():
    n_iterations = 10

    annealer = MH(
        {'density': nx.density},
        max_iterations=n_iterations,
        warm_start=True
    )

    annealer.spec_optimize(
        graph,
        graph
    )

    annealer.optimize(
        graph
    )

    assert annealer._state.iteration == 20

def test_callbacks():
    n_iterations = 10

    annealer = MH(
        functions,
        max_iterations=10,
        callbacks=[
            LoggingCallback(period=1)
        ]
    )

    annealer.spec_optimize(
        graph,
        graph
    )

    assert annealer.log__.shape[0] == n_iterations

    annealer.optimize(
        graph
    )

    assert annealer.log__.shape[0] == n_iterations

    annealer.warm_start = True
    annealer.optimize(graph)

    assert annealer.log__.shape[0] == 2 * n_iterations
















