'''A module to miniaturize graphs using a Metropolis-Hastings annealer

    - Author: David J. Butts
    - Date created: March 2024
    - Date Last Modified: September 2024
'''

from minigraphs.callback import LoggingCallback, EarlyStoppingCallback
import numpy as np
import networkx as nx
import pandas as pd
from copy import deepcopy
import scipy.sparse
from scipy.special import comb
import scipy
from sklearn.preprocessing import normalize
from sklearn.exceptions import NotFittedError
from typing import Callable 
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod
from collections import deque
from pydantic import BaseModel, validate_call
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

NX_DENSITY = lambda G: nx.density(G)
NX_CLUSTERING = lambda G: nx.average_clustering(G)
NX_ASSORTATIVITY = lambda G: nx.degree_assortativity_coefficient(G)

def sigmoid(x,x0,k):
    return 1 / (1 + np.exp(-k*(x-x0)))

def schedule_sigmoid(t_max,beta_max=1):
    k = 2*np.log(19)/t_max 
    t0 = t_max / 2
    
    return lambda t: sigmoid(t,t0,k) * beta_max

class Change(ABC):
    @abstractmethod
    def __init__(self,edges):
        pass 
    
    @abstractmethod
    def do(self,G):
        pass
    
    @abstractmethod
    def undo(self,G):
        pass
    
class Add(Change):
    def __init__(self,edge):
        self.edge = edge 
        
    def do(self,G):
        G.add_edge(self.edge[0],self.edge[1])
    
    def undo(self,G):
        G.remove_edge(self.edge[0],self.edge[1])
        
class Remove(Change):
    def __init__(self,edge):
        self.edge = edge 
        
    def do(self,G):
        G.remove_edge(self.edge[0],self.edge[1])
        
    def undo(self,G):
        G.add_edge(self.edge[0],self.edge[1])
        
class Switch(Change):
    def __init__(self,edges):
        self.edges = edges
    
    def do(self,G):
        old, new = self.edges
        G.remove_edge(old[0],old[1])
        G.add_edge(new[0],new[1])
        
    def undo(self,G):
        old, new = self.edges 
        G.remove_edge(new[0],new[1])
        G.add_edge(old[0],old[1])

class EvaluatorDict:
    '''Evaluator Class
    '''
    def __init__(self, dictionary: Dict[str, Callable], object):
        self.dictionary = dictionary
        self.object = object 
        
    def __getitem__(self, key):
        return self.dictionary[key](self.object)
    
    def __setitem__(self, key, func: Callable):
        self.dictionary[key] = func
    
    def keys(self):
        return self.dictionary.keys()
    
    def values(self):
        return [func(self.object) for func in self.dictionary.values()]
    
    def items(self):
        return {key: func(self.object) for key, func in self.dictionary.items()}.items()

@dataclass
class OptimizerState:
    step: int = 0
    iteration: int = 0
    beta: int = 0 
    metrics: List = field(default_factory=list)
    loss: List = field(default_factory=list)
    stop: bool = False 
    is_fitted: bool = False

class MH:
    """
    An MH-based annealer to miniaturize a graph.
    
    Attributes
    ----------
    schedule : callable
        The annealing schedule of the MH algorithm.

    weigths: dict of str to float
        Dictionary of associated with each target metric.

    graph_ : networkx.Graph
        The generated graph miniature.
    """
    @validate_call
    def __init__(
            self,
            metrics_functions: dict [str, Callable],
            schedule: float | Callable = float('inf'),
            metrics_weights: dict [str, float] = {},
            n_changes: int = 1,
            max_iterations: int = 1000,
            copy: bool = False,
            warm_start: bool = False,
            norm: str = 'l1',
            callbacks: List = []
        ):
        # Initialize attributes
        self.metrics_functions = metrics_functions 
        self.metrics_weights = metrics_weights
        self.n_changes = n_changes
        self.max_iterations = max_iterations
        self.schedule = schedule
        self.copy = copy
        self.warm_start = warm_start
        self.norm = norm
        self.callbacks = callbacks

        # Initialize internal state
        self._state = OptimizerState()

    def compute_graph_metrics(self, graph):
        '''Compute the metrics of a graph
        '''
        return {metric: func(graph) for metric, func in self.metrics_functions.items()}
    
    def compute_diff(self, metrics):
        '''Compute the difference in the metrics
        '''
        diff = {}

        for metric, value in metrics.items():
            try:
                diff[metric] = self.target_metrics_[metric] - value
            except TypeError as e:
                raise TypeError(f"Error: invalid scalar operation for metric {metric}") from e
            
        return diff

    def compute_loss(self, diff, p=1):
        '''Compute the induced loss of the graph from the differences
        '''
        loss = 0

        for metric, value in diff.items():
            loss += self._weights[metric] * abs(value) ** p

        return loss
        
    def _do(self):
        '''Implements changes in a graph
        '''
        self._actions = deque()
        
        # Propose changes
        for i in range(self.n_changes):
            # Propose change
            choose_action = True
            while choose_action:
                # Choose an action at random
                p = np.random.uniform()
                edges = list(nx.edges(self._miniature))
                non_edges = list(nx.non_edges(self._miniature))

                if (p < 0.25) and (len(non_edges) > 0):
                    # Add edge
                    idx = np.random.randint(0,len(non_edges))
                    
                    action = Add(non_edges[idx])
                    choose_action = False
                    
                elif (p < 0.5) and (len(edges) > 0):
                    # Remove edge
                    idx = np.random.randint(0,len(edges))
                    
                    action = Remove(edges[idx])
                    choose_action = False
                    
                elif (len(edges) > 0) and (len(non_edges) > 0):
                    # Switch edge
                    idx_edge = np.random.randint(0,len(edges))
                    idx_non_edge = np.random.randint(0,len(non_edges))
                
                    action = Switch((edges[idx_edge],non_edges[idx_non_edge]))
                    choose_action = False      
            
            # Implement change
            action.do(self._miniature)
            self._actions.append(action)   

    def _undo(self):
        '''Reverses changes made to the graph
        '''
        for _ in range(len(self._actions)):
            self._actions.pop().undo(self._miniature)

    def _accept(self, beta: float, loss: Tuple) -> bool:
        '''Accepts proposed change according to the Metropolis ratio
        '''
        return np.exp(beta * (loss[0] - loss[1])) >= np.random.uniform()
    
    def spec(self, target):
        '''Specifies the target
        '''
        if isinstance(target, Dict):
            # Validate metrics
            if self.metrics_functions.keys() != target.keys():
                raise KeyError("Annealer and target metrics do not coincide.")

            metrics = target
        else:
            metrics = self.compute_graph_metrics(target)

        self.target_metrics_ = metrics
        self._state.is_fitted = True

        return self

    def optimize(self, graph):
        '''Optimizes the given graph
        '''
        # Validate weights 
        if self.metrics_weights.keys() != self.metrics_functions.keys():
            self._weights = dict.fromkeys(self.metrics_functions.keys(), 1.0)
        else:
            self._weights = self.metrics_weights

        # Validate fitting
        if not self._state.is_fitted:
            raise NotFittedError("ERROR: Target metrics not yet specified.")
        
        # Validate schedule
        if isinstance(self.schedule, int | float):
            self._schedule = lambda t: self.schedule

        # Validate copy
        if self.copy:
            self._miniature = deepcopy(graph)
        else:
            self._miniature = graph

        # Validate norm
        match self.norm:
            case 'l1':
                p = 1
            case 'l2':
                p = 2
                
        # Compute metrics & loss
        self._state.metrics = [self.compute_graph_metrics(self._miniature), None]
        self._state.diff = [self.compute_diff(self._state.metrics[0]),  None]
        self._state.loss = [self.compute_loss(self._state.diff[0],p), None]

        # Initialize
        self._state.step = 0
        
        if not self.warm_start:
            # Reset iteration counter
            self._state.iteration = 0
            
            # Reset callbacks
            for callback in self.callbacks:
                callback._before_optimization(self._state)

        while (self._state.step < self.max_iterations) and (not self._state.stop):
            # Modify graph
            self._do()

            # Calculate acceptance parameters
            self._state.beta = self._schedule(self._state.iteration)

            self._state.metrics[1] = self.compute_graph_metrics(self._miniature)
            self._state.diff[1] = self.compute_diff(self._state.metrics[1])
            self._state.loss[1] = self.compute_loss(self._state.diff[1],p)
            
            # Check for change
            if self._accept(self._state.beta, self._state.loss):
                self._state.metrics[0] = self._state.metrics[1]
                self._state.diff[0] = self._state.diff[1]
                self._state.loss[0] = self._state.loss[1]
            else:
                # Undo Changes
                self._undo()     

            # Increase step & total number of iterations
            self._state.step += 1
            self._state.iteration += 1

            # Apply callbacks
            for callback in self.callbacks:
                callback._on_iteration_end(self._state)

        # End of optimization callbacks
        for callback in self.callbacks:
            callback._after_optimization(self._state)

            if isinstance(callback, LoggingCallback):
                self._log = callback

            if isinstance(callback, EarlyStoppingCallback):
                self.distance__ = callback.distance

        return self
    
    def spec_optimize(self, target, graph):
        '''Specifies target and optimizes the graph
        '''
        return self.spec(target).optimize(graph)

    @property 
    def metrics_keys(self):
        '''Report the names of the metrics
        '''
        return self.metrics_functions.keys()
    
    @property
    def weights__(self):
        '''Reports weights utilized
        '''
        return self._weights
    
    @property
    def miniature__(self):
        '''Reports the optimized miniature
        '''
        return self._miniature
    
    @property
    def metrics__(self):
        '''Reports the current metrics of the annealer
        '''
        return self._state.metrics[0]
    
    @property
    def loss__(self):
        '''Reports the current loss of the annealer
        '''
        return self._state.loss[0]
    
    @property
    def state__(self):
        '''Reports the current state of the annealer
        '''
        dictionary = {
            "Iteration": self._state.iteration,
            "Beta": self._state.beta,
            "Loss": self._state.loss[0],
            "Metrics": self._state.metrics[0]
        }

        return dictionary
    
    @property 
    def log__(self):
        return self._log.log
         
class CoarseNET:
    '''
    A class that implements the CoarseNET algorithm for an unweighted, 
    undirected graph.

    Attributes
    ----------
    G_coarse : networkx.Graph
        Coarsened graph.
    
    alpha : float
        Shrinkage factor
    '''
    
    def __init__(self,alpha: float,G: nx.Graph):
        '''
        Parameters
        ----------
        alpha : float
            Shrinkage factor.

        G : networkx.Graph
            Graph to coarsen.
        '''
        self.alpha = alpha
        self.G = deepcopy(G)
    
    @property
    def alpha(self):
        '''Shrinkage factor'''
        return self._alpha
    
    @alpha.setter
    def alpha(self,val):
        #TODO: Validate within range (0,1.0)
        self._alpha = val
        
    @property
    def G(self):
        '''Original Graph

        Returns
        -------
        G : networkx.Graph
            The original graph to miniaturize
        '''
        return self._G
    
    @G.setter
    def G(self,Graph: nx.Graph):
        '''
        Test
        Parameters
        ----------
        Graph : networkx.Graph
        '''
        #TODO: Validate strongly connected graph
        self._G = Graph
    
    @staticmethod
    def adjacency(G: nx.Graph):
        '''Returns the column-normalized adjacency matrix of
        a graph.

        Parameters
        ----------
        G : networkx.Graph
            A graph to construct the adjacency matrix.
        '''
        A = nx.to_scipy_sparse_array(G)
        A = A._asfptype()
        A = normalize(A,norm='l2',axis=0)
        
        return A
    
    @staticmethod
    def eigs(G: nx.Graph):
        '''Computes the dominant eigenvalue and eigenvectors
        associated with the adjacency matrix of a graph.

        Parameters
        ----------
        G : networkx.Graph
            Graph to calculate eigenvalue and eigenvectors.
        '''
        # Adjacency Matrix
        A = CoarseNET.adjacency(G)
        
        # Compute the first eigenvalue and right eigenvector
        lambda_, u_ = scipy.sparse.linalg.eigs(A,k=1)
        
        # Compute the left eigenvector
        _, v_= scipy.sparse.linalg.eigs(A.T,k=1)
                
        return np.real(lambda_)[0], np.real(np.squeeze(u_)), np.real(np.squeeze(v_))
    
    def __edge_score(self,edge):
        '''Calculates the score of a node pair
        '''
        u_a, u_b = self.u_[edge[0]], self.u_[edge[1]]
        v_a, v_b = self.v_[edge[0]], self.v_[edge[1]]
        
        prod = (self.lambda_-1)*(u_a+u_b)
        score = (-self.lambda_*(u_a*v_a+u_b*v_b) + v_a*prod + u_a*v_b + u_b*v_a) / (np.dot(self.v_,self.u_)-(u_a*v_a + u_b*v_b))  
    
        return score
        
    def __score(self):
        '''Calculates the score for all the edges in the graph
        '''
        # Initialize array of scores
        score = np.zeros(self.G_coarse_.number_of_edges())
        
        # Calculate score for every edge in the graph
        for i, edge in enumerate(self.G_coarse_.edges):
            score[i] = self.__edge_score(edge)
        
        return np.abs(score)
    
    def __contract(self,edge) -> bool:
        '''Updates graph by contracting nodes in the edge
        '''
        # Upack nodes
        u, v = edge
        left, right = self.nodes_coarse_[u], self.nodes_coarse_[v]
        
        contract = left != right
        if contract:
            # Merge nodes
            nx.contracted_nodes(self.G_coarse_,
                                left,
                                right,
                                self_loops=False,
                                copy=False)
            
            # Update node index in coarsened graph
            idx = self.nodes_coarse_ == right
            self.nodes_coarse_[idx] = left 
            self.nodes_removed_.append(right)
            
        return contract
        
    def coarsen(self) -> None:
        '''
        Coarsens the seed graph
        '''
        self.G_coarse_ = self._G.to_directed()
        n = self.G_coarse_.number_of_nodes()
        n_edges = self.G_coarse_.number_of_edges()
        n_reduced = int(self._alpha * n)
        
        # Compute the eigenvalue and eigenvectors
        self.lambda_, self.u_, self.v_ = CoarseNET.eigs(self.G_coarse_)
        
        # Arrays of nodes and edges
        self.nodes_coarse_ = np.arange(0,n,dtype=np.int32)
        self.nodes_removed_ = []
        edges = list(self.G_coarse_.edges)
        
        # Calculate sorting indices according to score
        score = self.__score()
        idx = np.argsort(score)
        
        contractions = 0
        i = 0
        while (contractions < n_reduced) and (i < n_edges):
            # Retrieve edge according to sorting
            edge = edges[idx[i]]
            
            # Contract edges
            contract = self.__contract(edge)
            
            if contract:
                contractions += 1
            
            i += 1

        # Add removed edges to the original graph
        #self.G_coarse_.add_nodes_from(self.nodes_removed_)
        
        
        
        
        
            
        
        