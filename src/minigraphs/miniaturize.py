'''A module to miniaturize graphs using a Metropolis-Hastings annealer

    - Author: David J. Butts
    - Date created: March 2024
    - Date Last Modified: September 2024
'''

import numpy as np
import networkx as nx
import pandas as pd
from copy import deepcopy
import scipy.sparse
from scipy.special import comb
import scipy
from sklearn.preprocessing import normalize
from typing import Callable 
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod
from collections import deque

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

    def __init__(self,
                 metrics: dict [str, Callable],
                 schedule: Callable | None = None, 
                 weights: dict [str, float ] = None,
                 n_changes: int = 1,
                 func_loss: Callable = None,
                 ):
        """
        Parameters
        ----------
        metrics : dict of str to callable
            Dictionary containing the functions used to evaluate the graph 
            metrics.

        schedule : callable
            A function that returns the value of the inverse temperature at
            every iteration.

        weights :  dict of str to float
            Dictionary containing the relative weights of every graph metric.

        n_changes : int
            Number of changes implemented at each iteration.

        func_loss : callable
            A callable that will take a list of weights and a list of differences
            in graph metrics and evaluate the loss.
        """

        # Store Annealing schedule
        if schedule is None:
            self.schedule = self.__schedule_adaptive
            self.__schedule_is_default = True
        else:
            self.schedule = schedule
            self.__schedule_is_default = False
        
        # Store metrics functions
        self._metrics = metrics 
    
        # Store weights
        if weights is None:
            # Weight metrics equally if no weights are specified
            self._weights = {key: 1.0 for key in self._metrics}
        else:
            self._weights = weights
        
        # Check for matching keys
        if self._metrics.keys() != self._weights.keys():
            raise ValueError("Specified weights don't match corresponding metrics")
        
        # Store number of changes per step
        self.n_changes = n_changes
        self.__window_size = 10
        
        # Store loss function
        if func_loss is None:
            self.func_loss = lambda weights, metrics_diff: np.sum(weights * np.abs(metrics_diff))
        else:
            self.func_loss = func_loss
        
    @property
    def metrics(self) -> list[str]:
        """
        Names of metrics to target
        """
        return list(self._metrics.keys())
    
    @property
    def weights(self) -> list[str]:
        """
        Names of metrics associated with weights
        """
        return list(self._weights.keys())
    
    @property
    def trajectories_(self) -> pd.DataFrame:
        """
        DataFrame containing the trajectories of associated
        with the miniaturization procedure
        """
        names = ['Step','Beta','Energy'] + list(self._targets_names)
        df = pd.DataFrame(self._trajectories_,columns=names)
        df.set_index('Step',inplace=True)
        
        return df
    
    def __schedule_adaptive(self,step):
        '''Provides an adaptive scheduling
        '''
        f = 1.0
        if step >= self.__window_size:
            # Calculate current acceptance rate
            rate = self.__window.sum() / self.__window_size
            
            # Adjust rate
            if rate > 0.40:
                f = 1.01
            elif rate < 0.20:
                f = 0.99
                
        return self.beta * f
    
    def __stop_convergence(self,step):
        '''Provide a convergence criterion based on the proximity to the metrics
        '''
        window_size = 100
        
        if step <= window_size:
            flag = False
        else:
            flag = self.__E0 < self.__epsilon
            
        return flag
        
                
        
    def __get_metrics(self):
        '''Calculates the metrics of a graph
        '''
        metrics = []
        
        for name in self._targets_names:
            # Calculate metric
            metric = self._metrics[name](self.graph_)
            
            if not isinstance(metric, (int,float)):
                raise TypeError(f"Metric function {name} returned a non-scalar value")
            else:
                metrics.append(metric)
                
        return metrics
            
    def __energy(self, metrics):
        '''Energy of the graph with respect to target metrics
        '''
        diff = self._targets - metrics
        
        energy = self.func_loss(self._weights_arr,diff)
        
        return energy
        
    def __make_change(self) -> None:
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
                edges = list(nx.edges(self.graph_))
                non_edges = list(nx.non_edges(self.graph_))

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
            action.do(self.graph_)
            self._actions.append(action)          
    
    def __accept_change(self,E0: float, E1:float) -> bool:
        '''Accepts proposed change according to the Metropolis ratio
        '''
        return np.exp((E0-E1)*self.beta) >= np.random.uniform()

            
    def transform(self, 
                  graph_seed: nx.Graph, 
                  targets: dict[str,float],
                  n_iterations: int = None,
                  epsilon: float = None,
                  beta: float = None,
                  verbose: bool = False) -> None:
        '''
        Miniaturizes seed graph

        Parameters
        ----------
        graph_seed: nx.Graph
            The initial graph to optimize.

        targets: dict of str to float
            Dictionary containing the graph metrics to target.

        n_iterations: int
            Number of iterations the annealer will be run for.
        
        epsilon: float
            Tolerance value to determine metrics convergence. 

        beta: float
            Initial inverse-temperature to be used in the automatic
            scheduling.

        verbose: bool
            Flat indicating whether or not to print progress.
        '''
        if (n_iterations is None) and (epsilon is None):
            raise ValueError("Exactly one stopping criterion must be provided")
        elif epsilon is None:
            stop = lambda step: step >= n_iterations
        elif n_iterations is None:
            self.__epsilon = epsilon
            stop = self.__stop_convergence
        
        # Verify matching keys
        self._targets_names = set(self.metrics).intersection(set(targets.keys()))
        self._n_states = len(self._targets_names)
        if self._n_states != 0:
            # Initialize internal variables
            self._weights_arr = np.array([self._weights[key] for key in self._targets_names])
            self._targets = np.array([targets[key] for key in self._targets_names])
        else:
            raise LookupError(f"No valid targets specified for annealer with metrics {self.metrics}\n")
        
        if (not self.__schedule_is_default) and (beta is not None):
            print("Schedule provided - ignoring initial value for beta")
        elif beta is not None:
            self.beta = beta 
        elif self.__schedule_is_default:
            raise ValueError("Initial beta not provided for adaptive scheduling")
        
        # Create window to keep track of acceptance
        self.__window = np.zeros(self.__window_size,dtype=bool)
        self.__idx_window = 0

        # Initialize graph
        self.graph_ = deepcopy(graph_seed)
        
        # Calculate graph metrics and energy
        self.__m0 = self.__get_metrics()
        self.__E0 = self.__energy(self.__m0)
        
        # Initialize trajectories
        size_history = n_iterations or 10000
        self._trajectories_ = np.zeros((size_history,self._n_states+3))
        
        #  Begin optimization
        self.__idx_history = 0
        step = 0
        while not stop(step):
            if (verbose is True) and (n_iterations is not None):
                print(f"Iteration {step+1}/{n_iterations}\n")
                
            # Obtain temperature according to schedule
            self.beta = self.schedule(step)
                
            # Change the graph and calculate energy
            self.__make_change()
            m1 = self.__get_metrics()
            E1 = self.__energy(m1)
            
            # Check for change
            if self.__accept_change(self.__E0, E1):
                # Update metrics and energy
                self.__m0 = m1
                self.__E0 = E1
                
                self.__window[self.__idx_window] = True
            else:
                # Reverse changes
                for i in range(len(self._actions)):
                    self._actions.pop().undo(self.graph_)
                    
                self.__window[self.__idx_window] = False
            
            # Update window index
            self.__idx_window = (self.__idx_window + 1) % self.__window_size
            
            # Record state
            self._trajectories_[self.__idx_history][0] = step
            self._trajectories_[self.__idx_history][1] = self.beta
            self._trajectories_[self.__idx_history][2] = self.__E0
            self._trajectories_[self.__idx_history][3:] = self.__m0
            
            # Update indices
            self.__idx_history = (self.__idx_history + 1) % size_history
            step += 1
            
        # Store only required iterations
        if step <= size_history:
            self._trajectories_ = self._trajectories_[0:step]
        else:
            # Slice array
            stack  = (
                self._trajectories_[self.__idx_history:],
                self._trajectories_[0:self.__idx_history]
            )
            
            self._trajectories_ = np.vstack(stack)
            
        
    
    @staticmethod
    def plot_trajectories(data,targets=None):
        # Number of trajectories
        trajectories = data.columns
        n_trajectories = len(trajectories)
        
        # Instantiate figure and axes
        fig, axes = plt.subplots(n_trajectories,1,dpi=300,figsize=(5,n_trajectories))
        
        for i,trajectory in enumerate(trajectories):
            # Plot trajectory
            axes[i].plot(data[trajectory],linewidth=1.0)
            
            # Plot targets if specified
            if (trajectory != 'Beta') and (trajectory != 'Energy') and (not targets is None):
                axes[i].axhline(targets[trajectory],linestyle='--',linewidth=0.5,color='red')
                
            # Label axes
            axes[i].set_ylabel(trajectory)
            
            if i != (n_trajectories-1):
                axes[i].set_xticklabels([])
                
        axes[n_trajectories-1].set_xlabel("Step")
        
        return axes
            
            
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
        
        
        
        
        
            
        
        