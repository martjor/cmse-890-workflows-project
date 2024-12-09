#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import networkx as nx
from minigraphs import simulation as sim 
from minigraphs.utils import dispatch
import numpy as np
import matplotlib.pyplot as plt
import click
import pandas as pd
import yaml
from inspect import signature
import os
from scipy.sparse import load_npz
import glob
from itertools import combinations

class io:
    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
        
    @staticmethod
    def load_graph(path):
        graph = nx.from_scipy_sparse_array(load_npz(path))
        
        return graph
        
    @staticmethod
    def get_arguments(func):
        # Extract the parameter names from the signature
        return [param.name for param in signature(func).parameters.values()]

    @staticmethod
    def mkdir(output_dir):
        '''Creates output directory if it doesn't exist already
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    

def sample(parameter,
           inputs,
           n_graphs,
           n_vertices):
    '''Generates a an ensamble of graphs according to the indicated distribution
    '''
    def get_error(metric,target):
        '''Calculates the error in a generated metric
        '''
        return np.abs(metric - target) / target * 100
    
    inputs = eval(inputs)
    
    # Dictionary of generators
    generators = {
        "density": lambda n,p: nx.erdos_renyi_graph(n,p),
        "clustering": lambda n,p: nx.erdos_renyi_graph(n,p)
    }
    
    metrics_dict = {
        "density": lambda G: nx.density(G),
        "clustering": lambda G: nx.average_clustering(G),
        "assortativity": lambda G: nx.degree_assortativity_coefficient(G)
    }
    
    # Select generator according to parameter
    generator = generators[parameter]

    # Allocate memory for data
    n_inputs = inputs.shape[0]
    graphs = [0] * n_inputs
    columns = list(metrics_dict.keys()) + ["error"]
    n_columns = len(columns)
    
    graphs = np.empty((n_inputs,n_graphs),dtype=object)
    parameters = np.zeros((n_inputs,n_graphs,n_columns))
    
    for i, p in enumerate(inputs):
        for j in range(n_graphs):
            # Generate graph
            graphs[i,j] = generator(n_vertices,p)
            
            # Calculate metrics
            metrics = {key: func(graphs[i,j]) for key, func in metrics_dict.items()}
            error = get_error(metrics[parameter],p)
            parameters[i,j,:] = list(metrics.values()) + [error]
    
    # Obtain data and metadata
    graphs = graphs.reshape(-1,1).squeeze()
    parameters = parameters.reshape(-1,n_columns).squeeze()
    parameters = pd.DataFrame(parameters,columns=columns)
    parameters["parameter"] = parameter
    
    return graphs, parameters

class Database:
    def __init__(self):
        self.DATA_DIR = "/mnt/home/martjor/repos/dev_pt/workflows/sensitivities/data"
        self.GRAPHS_DIR = os.path.join(self.DATA_DIR,'samples')
        
    @property
    def graphs(self):
        pattern = os.path.join(self.GRAPHS_DIR,'**/*.csv')
        files = glob.glob(pattern,recursive=True)
        
        table = []
        for file in files:
            df = pd.read_csv(file)
            
            table.append(df)
            
        table = pd.concat(table,ignore_index=True)
        return table
        
class Characterization:
    '''Characterizes a set of graphs
    '''
    def __init__(self, graphs: list[nx.Graph]):
        self.graphs = graphs 
        self.n_graphs = len(graphs)
    
    @staticmethod
    
    
    def distance_distribution(G):
        '''Calculates the distributions of pairwise distances
        '''
        paths = dict(nx.all_pairs_shortest_path(G))
        dist = {}
        
        # Obtain histogram
        for node_a in range(G.number_of_nodes()):
            # Iterate through neighbors
            for path in paths[node_a].values():
                # Calculate path length
                length = len(path) - 1
                
                # Avoid self-loops
                if length != 0: 
                    # Add length to dictionary
                    if not length in dist:
                        dist[length] = 1
                    else:
                        dist[length] += 1
        
        # Obtain largest distance
        girth = max(dist.keys())
        
        # Allocate memory for the distribution
        dist_arr = np.zeros((girth,2))
        dist_arr[:,0] = np.arange(1,girth+1)
        dist_arr[:,1] = np.array(list(dist.values()))
            
        # Normalize distribution
        dist_arr[:,1] /= dist_arr[:,1].sum() * 2
        
        return dist_arr
        
    @staticmethod
    def moment(dist,p):
        '''Calculates the pth momenth of a distribution
        '''
        moment = ((dist[:,0] ** p) * dist[:,1]).sum()
        return moment
    
    def report(self):
        '''Generates a report of characteristics
        '''
        columns = ['degree^1','degree^2','dist^1','dist^2']
        dist_degree = [0] * self.n_graphs
        dist_paths = [0] * self.n_graphs
        metadata = np.zeros((self.n_graphs,len(columns)))
        
        for i, graph in enumerate(self.graphs):
            # Calculate distributions
            dist_degree[i] = Characterization.degree_distribution(graph)
            dist_paths[i] = Characterization.distance_distribution(graph)
            
            # Calculate moments
            moments = [Characterization.moment(dist_degree[i],1),
                       Characterization.moment(dist_degree[i],2),
                       Characterization.moment(dist_paths[i],1),
                       Characterization.moment(dist_paths[i],2)]
            
            metadata[i,:] = moments
        
        # Convert metadata into a dataframe
        metadata = pd.DataFrame(metadata,columns=columns)
        
        return dist_degree, dist_paths, metadata
    
def simulate_sir(adjacencies,
                 tau,
                 gamma,
                 n_iterations,
                 n_runs):
    '''Simulates the sir model on the specified graphs
    '''
    n_graphs = len(adjacencies)
 
    # Instantiate sir object
    sir = sim.Sir(tau,gamma)
    sim_results = [0] * n_graphs

    # Simulate on each graph
    for adjacency in adjacency:
        # Instantiate simulation object
        simulation = sim.Simulation(adjacency)
        
        # Allocate memory
        trajectories = np.zeros(n_runs,n_iterations,3)

        for i in n_runs:
            # Simulate
            simulation.run(sir, n_steps)
        

    



