import numpy as np
import pandas as pd
import abc
from sklearn.preprocessing import normalize

class Simulation:
    def __init__(self,adjacency):
        '''Initializes the simulation object on an adjacency matrix
        '''
        self._A = adjacency
        
    @property
    def trajectories_(self):
        return self._trajectories_
    
    @property
    def trajectories_df_(self):
        df = pd.DataFrame(self._trajectories_,columns=self.columns_)
        return df
    
    @property
    def n_agents(self):
        return self._A.shape[0]
    
    @property
    def adjacency(self):
        return self._A.copy()
        
    def run(self,model, n_steps: int):
        # Check model formulation
        n_trajectories, self.columns_ = model.spec(self._A)
        
        # Initialize array of trajectories
        self._trajectories_ = np.zeros((n_steps,n_trajectories))
        
        # Process adjacency matrix
        A = model.process(self._A)
        
        # Set initial conditions and calculate associated trajectories
        states_buffer = np.zeros((2,self.n_agents))
        states_buffer[0,:] = model.start(A)
        self._trajectories_[0] = model.trajectories(A,states_buffer[0,:])
        
        # Run simulation
        idx_old = 0
        idx_new = 1
        for i in range(1,n_steps):
            # Update states
            model.step(A,states_buffer[idx_old,:],states_buffer[idx_new,:])
            
            # Calculate trajectories
            self._trajectories_[i] = model.trajectories(A,states_buffer[idx_new,:])
            
            # Update buffer index
            idx_old = (idx_old + 1) % 2
            idx_new = (idx_new + 1) % 2
            
class DeGroot:        
    def process(self,A):
        return normalize(A,axis=0,norm='l1')
    
    def spec(self,A):
        n_agents = A.shape[0]
        return n_agents, np.arange(n_agents,dtype=np.int32)
    
    def start(self,A):
        '''Initializes the state of each agent
        '''
        n_agents = A.shape[0]
        state = np.random.rand(n_agents)
        return state 
    
    def trajectories(self,A,state):
        '''Computes the trajectories of the agents
        '''
        return state
    
    def step(self,A,states_old,states_new):
        states_new[:] = A.dot(states_old)
        
              
class Sir:
    def __init__(self,tau: float,gamma: float, n_infected: int = 1):
        '''Initialilzes the sir simulation object
        '''
        #TODO: Implement check for bounds 
        self.tau = tau 
        self.gamma = gamma
        self.n_infected = n_infected
        
    def spec(self,A):
        return 3, ('S','I','R')
        
    def process(self,A):
        return A
    
    def start(self,A):
        '''Initializes the state of each agent in the network
        '''
        n_agents = A.shape[0]
        
        # Initialize agents as susceptible
        states = np.zeros(n_agents,dtype=np.int32)
        
        # Randomly select agents to be infected
        idx = np.random.choice(n_agents,self.n_infected,replace=False)
        states[idx] = 1
        
        return states
    
    def trajectories(self,A,states):
        '''Calculates the simulation trajectories
        '''
        return np.sum(states[:,np.newaxis] == np.array([0,1,2]),axis=0) / states.shape[0]
    
    def step(self,A,states_old,states_new):
        '''Updates the states of the agents
        '''
        n_agents = A.shape[0]
        self.beta = self.tau / n_agents
        if self.beta > 1.0:
            raise ValueError("Beta must be smaller or equal than 1.0")
        
        for i in range(n_agents):
            # Check current agent's state
            state = states_old[i]
            
            if state == 0:
                # Obtain neighbors of the ith agent
                row = A._getrow(i)
                
                # Count number of infected neighbors
                n_infected = int((states_old[row.indices] == 1).sum())
                
                # Attempt infection according to the binomial distribution
                if np.random.rand() <= self.__p_infection(n_infected):
                    states_new[i] = 1
                else:
                    states_new[i] = 0
                    
            elif state == 1:
                # Recover infected agent with probability gamma
                if np.random.rand() <= self.gamma:
                    states_new[i] = 2
                else:
                    states_new[i] = 1
                    
            else:
                # Recovered individuals remain recovered
                states_new[i] = 2
                
    def __p_infection(self,n_contacts: int) -> float:
            p = n_contacts and (n_contacts * self.beta * ((1-self.beta)**(n_contacts-1)))
            return p
            
                
            
            
        
        
        
            
        
    