#!/usr/bin/env python3
from minigraphs.miniaturize import MH
from numpy import log
import pandas as pd 
import networkx as nx
import sys
import os
import yaml
from utils import StreamToLogger
import logging
'''Calculates the parameters for the specified graph
'''

# Get the log file from Snakemake
log_file = snakemake.log[0]

# Configure logging to write to the Snakemake log file
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,  # Capture all logs (DEBUG and above)
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Replace stdout and stderr with the logger
sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

def weights(metrics_file,
           params_file,
           shrinkage,
           n_changes,
           n_samples,
           n_iterations):
    
    # Retrieve Graph Metrics
    with open(metrics_file) as file:
        metrics = yaml.safe_load(file)
        metrics_target = {key:metrics[key] for key in ['density','assortativity_norm','clustering']}

    # Specify Metric funcions
    funcs_metrics = {
        'density': nx.density,
        'assortativity_norm': lambda G: (nx.degree_assortativity_coefficient(G)+1)/2,
        'clustering': nx.average_clustering
    }

    # Calculate miniature size
    try:
        n_vertices = int(metrics['n_vertices'] * (1-shrinkage))
        
        if (n_vertices < 1):
            raise ValueError
        
    except ValueError:
        print("Error: Invalid number of vertices in the miniature")
    
    print(f"Calculating parameters for graph at graph at {metrics_file}")
    print(f"\t - Size: {n_vertices} nodes ({shrinkage:.02f}% miniaturization)")
    print(f"\t - Number iterations per sample: {n_iterations}")
    print(f"\t - Number of samples: {n_samples}\n")

    # Calculate weights
    params = []
    for i in range(n_samples):
        print(f"Sweep {i+1}/{n_samples}")
        
        # Construct replica
        replica = MH(funcs_metrics,
                     schedule=lambda beta:0,
                     n_changes=n_changes)
        
        # Transform ER graph
        G = nx.erdos_renyi_graph(n_vertices,metrics_target['density'])
        replica.transform(G,
                          metrics_target,
                          n_iterations=n_iterations)

        # Retrieve trajectories
        df = replica.trajectories_
        weights = dict(1/df[replica.metrics].diff().abs().mean())
        
        print("Weights:")
        print(weights)

        # Calculate optimal beta
        replica = MH(funcs_metrics,
                     schedule=lambda beta:0,
                     n_changes=n_changes,
                     weights=weights)
        
        # Transform ER graph
        G = nx.erdos_renyi_graph(n_vertices,metrics_target['density'])
        replica.transform(G,
                          metrics_target,
                          n_iterations=n_iterations)

        df = replica.trajectories_

        beta = -log(0.23) * 1/df['Energy'].diff().abs().mean()

        print(f"Beta: {beta}\n")

        params.append([beta] + list(weights.values()))

    params = pd.DataFrame(params,columns=['beta'] + list(weights.keys()))
    print("Measeured parameters:")
    print(params,f"\n")

    params = params.mean()
    print("Final parameters:")
    print(params)

    # Save to yaml
    with open(params_file,'w') as file:
        yaml.dump(params.to_dict(),file,default_flow_style=False)
        
weights(snakemake.input[0],
        snakemake.output[0],
        0.9,
        10,
        30,
        100)
    