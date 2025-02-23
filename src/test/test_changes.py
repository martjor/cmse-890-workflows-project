import networkx as nx
import random 
import minigraphs.changes as changes

random.seed(42)

def test_Add():
    graph = nx.Graph()
    graph.add_nodes_from([0,1,2])

    add = changes.Add()

    add.apply(graph)
    assert graph.number_of_edges() == 1

    add.apply(graph)
    assert graph.number_of_edges() == 2

    add.apply(graph)
    assert graph.number_of_edges() == 3

    assert add.apply(graph) is False

    add.revert(graph)
    assert graph.number_of_edges() == 2


def test_Remove():
    graph = nx.complete_graph(3)

    remove = changes.Remove()

    remove.apply(graph)
    assert graph.number_of_edges() == 2

    remove.apply(graph)
    assert graph.number_of_edges() == 1

    remove.apply(graph)
    assert graph.number_of_edges() == 0

    assert remove.apply(graph) is False

    remove.revert(graph)
    assert graph.number_of_edges() == 1

    remove.revert(graph)
    assert graph.number_of_edges() == 2

def test_switch():
    graph = nx.Graph()
    graph.add_nodes_from([0,1,2])
    graph.add_edge(0,1)

    switch = changes.Switch()

    switch.apply(graph)
    assert graph.number_of_edges() == 1

    switch.apply(graph)
    assert graph.number_of_edges() == 1

    switch.revert(graph)
    assert graph.number_of_edges() == 1

    switch.revert(graph)
    assert graph.number_of_edges() == 1

    assert (0,1) == list(nx.edges(graph))[0]

    try: 
        switch.revert(graph)
    except Exception as e:
        assert isinstance(e, IndexError)


    
