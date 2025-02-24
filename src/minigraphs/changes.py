from abc import ABC, abstractmethod
from collections import deque
import random
import networkx as nx
from dataclasses import dataclass
from typing import Tuple
from math import isclose

def add_random_edge(graph):
    '''Adds a random edge
    '''
    edge = ()
    n_nodes = graph.number_of_nodes()
    n_edges = int(n_nodes * (n_nodes-1) / 2 - graph.number_of_edges())

    # Add edge if empty edges available
    if n_edges > 0:
        choice = random.randint(0, n_edges-1)

        for i in range(n_edges):
            if i == choice:
                edge = next(nx.non_edges(graph))
                graph.add_edge(*edge)
                return edge

    return edge

def remove_random_edge(graph):
    '''Removes an edge randomly
    '''
    edge = ()
    n_edges = graph.number_of_edges()

    # Remove edge if graph contains edges
    if n_edges > 0:
        i = 0
        choice = random.randint(0, n_edges-1)

        for i, edge in enumerate(nx.edges(graph)):
            if i == choice:
                graph.remove_edge(*edge)
                return edge

    return edge

@dataclass
class Change(ABC):
    edges: Tuple

    @abstractmethod
    def __init__(self, graph):
        '''Sublcasses must initialize `edges`.'''
        pass 

    @abstractmethod
    def revert(self, graph):
        '''Reverses the previous change to the graph'''
        pass

class Add(Change):
    def __init__(self, graph):
        edge = add_random_edge(graph)
        self.edges = (edge,) if edge else ()
    
    def revert(self, graph):
        if self.edges:
            graph.remove_edge(*self.edges[0])

class Remove(Change):
    def __init__(self, graph):
        edge = remove_random_edge(graph)
        self.edges = (edge,) if edge else()

    def revert(self, graph):
        if self.edges:
            graph.add_edge(*self.edges[0])

class Switch(Change):
    def __init__(self, graph):
        edges = (remove_random_edge(graph), add_random_edge(graph))
        #print(edges)
        self.edges = edges if (all(edges)) and (edges[0] != edges[1]) else ()

    def revert(self, graph):
        if self.edges:
            graph.remove_edge(*self.edges[1])
            graph.add_edge(*self.edges[0])

class ChangeSampler:
    def __init__(self, changes):
        self.names = list()
        self.changes = list()
        self.cdf = list()

        names_count = {}
        count = 0
        sum = 0
        for name, change, probability in changes:
            # Validate name
            if name in names_count:
                names_count[name] += 1
                name = name + f"_{names_count[name]}"
                count += 1
            else:
                names_count[name] = 0
            
            self.names.append(name)

            # Validate probability
            if 0 <= probability < 1:
                sum += probability
                self.cdf.append(sum)
            else:
                raise ValueError(f"ERROR: invalid probability for change '{name}': {probability}")
            
            self.changes.append(change)

        # Validate total probability
        if not isclose(sum, 1.0, abs_tol=1e-8) or (sum > 1.0):
            raise ValueError(f"ERROR: total probability must add up to 1.0 (got {sum:.2f})")
        
    def __call__(self, graph):
        '''Implements a change according to the probability distribution
        '''
        probability = random.random()

        for i, threshold in enumerate(self.cdf):
            if probability < threshold:
                return self.changes[i](graph)
            

            
        
            
            



        


            
    
    
        
        
        








