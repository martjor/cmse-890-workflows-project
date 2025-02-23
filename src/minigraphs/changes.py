from abc import ABC, abstractmethod
from collections import deque
import random
import networkx as nx

class Change(ABC):
    def __init__(self):
        self.edges = deque()
        self.n_changes = 0

    @abstractmethod
    def apply(self, graph):
        pass
    
    @abstractmethod
    def revert(self, graph):
        pass
    
class Add(Change):
    def apply(self, graph):
        '''Adds a random edge
        '''
        n_nodes = graph.number_of_nodes()
        n_edges = int(n_nodes * (n_nodes-1) / 2 - graph.number_of_edges())

        if n_edges > 0:
            choice = random.randint(0, n_edges-1)

            for i, edge in enumerate(nx.non_edges(graph)):
                if i == choice:
                    graph.add_edge(*edge)
                    self.edges.append(edge)
                    self.n_changes +=1

            return True
        else: 
            return False
        
    def revert(self, graph):
        '''Removes the previously added edge
        '''
        edge = self.edges.pop()
        graph.remove_edge(*edge)
        self.n_changes -= 1
        
class Remove(Change):
    def apply(self, graph):
        '''Removes an edge randomly
        '''
        n_edges = graph.number_of_edges()

        if n_edges > 0:
            choice = random.randint(0, n_edges-1)

            for i, edge in enumerate(nx.edges(graph)):
                if i == choice:
                    graph.remove_edge(*edge)
                    self.edges.append(edge)
                    self.n_changes += 1

            return True
        else: 
            return False
    
    def revert(self, graph):
        '''Adds the previously removed edge
        '''
        edge = self.edges.pop()
        graph.add_edge(*edge)
        self.n_changes -= 1

class Switch:
    def __init__(self):
        self._add = Add()
        self._remove = Remove()
        self._n_changes = 0

    def apply(self, graph):
        '''Switches an edge
        '''

        if (self._remove.apply(graph)) and (self._add.apply(graph)):
            self._n_changes += 1

            return True 
        else: 
            return False

    def revert(self, graph):
        '''Reverts edge switches
        '''
        self._add.revert(graph)
        self._remove.revert(graph)








