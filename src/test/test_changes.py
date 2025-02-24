import networkx as nx
import random 
from minigraphs.changes import Add, Remove, Switch, ChangeSampler
random.seed(42)

def test_add():
    graph = nx.Graph()
    graph.add_nodes_from([0,1,2])

    changes = []
    for _ in range(3):
        changes.append(Add(graph))

    assert graph.number_of_edges() == 3

    assert not Add(graph).edges 

    for change in reversed(changes):
        change.revert(graph)

    assert graph.number_of_edges() == 0

def test_remove():
    graph = nx.complete_graph(3)

    changes = []
    for _ in range(3):
        changes.append(Remove(graph))

    assert graph.number_of_edges() == 0

    assert not Remove(graph).edges

    for change in reversed(changes):
        change.revert(graph)

    assert graph.number_of_edges() == 3

def test_switch():
    graph = nx.Graph()
    graph.add_nodes_from([0,1,2])

    assert not Switch(graph).edges

    edges = ((1,0),(2,1))
    graph.add_edges_from(edges)

    changes = []
    for _ in range(10):
        change = Switch(graph)
        assert graph.number_of_edges() == 2

        edges = change.edges
        if edges:
            assert edges[0] != edges[1]

        changes.append(change)

    for change in reversed(changes):
        change.revert(graph)

    assert list(nx.edges(graph)) == [(0,1),(1,2)]

def test_sampler():
    change = ChangeSampler(
        changes=[
            ('add', Add, 1/3),
            ('remove', Remove, 1/3),
            ('switch', Switch, 1/3)
        ]
    )

    graph = nx.Graph()
    graph.add_nodes_from(range(10))

    changes = []
    for _ in range(30):
        changes.append(change(graph))

    for change in reversed(changes):
        change.revert(graph)

    assert not list(nx.edges(graph))


    
