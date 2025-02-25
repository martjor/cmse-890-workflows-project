'''A module to miniaturize graphs using a Metropolis-Hastings annealer

    - Author: David J. Butts
    - Date created: March 2024
    - Date Last Modified: September 2024
'''

from minigraphs.callback import LoggingCallback, EarlyStoppingCallback
from minigraphs.changes import Add, Remove, Switch, ChangeSampler
import numpy as np
from copy import deepcopy
from sklearn.exceptions import NotFittedError
from typing import Callable 
from typing import Dict, Tuple, List
from dataclasses import dataclass, field
from inspect import isclass

@dataclass
class OptimizerState:
    step: int = 0
    iteration: int = 0
    beta: int = 0 
    metrics: List = field(default_factory=list)
    loss: List = field(default_factory=list)
    stop: bool = False 
    is_fitted: bool = False

balanced_sampler = ChangeSampler(
    changes=[
        ('add', Add, 1/3),
        ('remove', Remove, 1/3),
        ('switch', Switch, 1/3)
    ]
)

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
    def __init__(
            self,
            metrics_functions: dict [str, Callable],
            schedule: float | Callable = float('inf'),
            metrics_weights: dict [str, float] = {},
            change = balanced_sampler,
            max_iterations: int = 1000,
            copy: bool = False,
            warm_start: bool = False,
            norm: str = 'l1',
            callbacks: List = []
        ):
        # Initialize attributes
        self.metrics_functions = metrics_functions 
        self.metrics_weights = metrics_weights
        self.change = change
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
        elif callable(self.schedule):
            self._schedule = self.schedule

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
                
        # Validate callbacks
        for callback in self.callbacks:
            if isclass(callback):
                raise TypeError("ERROR: Callbacks should be instances not classes.")
                
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
            # Apply change to graph
            change = self.change(self._miniature)

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
                # Revert change
                change.revert(self._miniature)

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
    def beta__(self):
        '''Reports the current inverse temperature of the annealer
        '''
        return self._state.beta
    
    @property
    def state__(self):
        '''Reports the current state of the annealer
        '''
        dictionary = {
            "iteration": self._state.iteration,
            "beta": self._state.beta,
            "loss": self._state.loss[0],
        }

        for metric, value in self._state.metrics.items():
            dictionary[f"m_{metric}"] = value

        return dictionary
    
    @property 
    def log__(self):
        return self._log.log
         
        
        
        
        
        
            
        
        