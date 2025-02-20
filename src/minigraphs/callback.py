from collections import deque
from math import sqrt
import pandas as pd

class Callback:
    def _before_optimization(self, _state):
        '''Resets the callback.
        '''
    
    def _after_optimization(self, _state):
        '''Called before the optimization process begins
        '''
        pass

    def _on_iteration_end(self, _state):
        '''Called at the end of each iteration.
        '''
        pass 

    

class LoggingCallback(Callback):
    def __init__(self, period=1):
        self.period = period
        self._log = deque()
        self._metrics = []

    def _before_optimization(self, _self):
        '''Reset the logger
        '''
        self._log = deque()
        self._metrics = []

    def _after_optimization(self, _state):
        '''Called at the end of the optimization
        '''
        # Obtain keys header
        self._metrics = list(_state.metrics[0].keys())

    def _on_iteration_end(self, _state):
        '''Called at the end of each iteration
        '''
        if (_state.iteration % self.period == 0):
            self._log.append([
                _state.iteration, 
                _state.beta,
                _state.loss[0],
                *_state.metrics[0].values()
                ]
            )

    @property
    def log(self):
        '''Returns a DataFrame containing the log of the optimizer
        '''
        columns = [
            'iteration',
            'beta',
            'loss',
            *[f"m_{metric}" for metric in self._metrics]
        ]

        df = pd.DataFrame(
            list(self._log),
            columns=columns,
        )

        return df

class EarlyStoppingCallback(Callback):
    def __init__(self, tol=0.05, period=1):
        self.period = period
        self.tol = tol
        self.distance = float('inf')

    def _before_optimization(self, _state):
        _state.stop = False

    def _on_iteration_end(self, _state):
        '''Checks for distance on iteration and stops
        '''
        if (_state.iteration % self.period == 0):
            self.distance = 0

            for value in _state.diff[0].values():
                self.distance += value ** 2

            self.distance = sqrt(self.distance)

            _state.stop = self.distance < self.tol

        

        
