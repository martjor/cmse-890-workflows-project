#!/usr/bin/env python3
from minigraphs.reduction import MH
from minigraphs.callback import LoggingCallback
from scripts.reduction import pt_setup as setup

from math import log, inf, nan 
import pandas as pd 
import networkx as nx
import sys
import yaml
from scripts.utils.io import StreamToLogger
import logging
from scripts.reduction.pt_setup import DICT_METRICS_FUNCS
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

def weights(targets,
            metrics_file,
            params_file,
            shrinkage,
            n_samples,
            n_iterations):
    '''
    '''
    ### VALIDATE INPUTS ###
    with open(metrics_file) as file:
        graph_metrics = yaml.safe_load(file)

    # Validate shrinkage factor
    n_vertices = int(graph_metrics['n_nodes'] * (1-shrinkage))
    if (n_vertices < 1):
            raise ValueError
    
    metrics={}
    functions={}
    for target in targets:
        metrics[target] = graph_metrics[target]
        functions[target] = DICT_METRICS_FUNCS[target]

    # Instantiate Optimizer
    annealer = MH(
         functions,
         copy=True,
         warm_start=False,
         schedule=0,
         callbacks=[LoggingCallback()],
         max_iterations=n_iterations
    )

    print(f"Calculating parameters for graph at graph at {metrics_file}")
    print(f"\t - Size: {n_vertices} nodes ({shrinkage * 100:.02f}% miniaturization)")
    print(f"\t - Number iterations per sample: {n_iterations}")
    print(f"\t - Number of samples: {n_samples}\n")

    # Calculate weights
    parameters = []
    for i in range(n_samples):
        print(f"Sweep {i+1}/{n_samples}...")

        # Reset weights
        annealer.metrics_weights = {metric: 1.0 for metric in metrics.keys()}
        
        # Initiate optimization
        graph = nx.erdos_renyi_graph(n_vertices, graph_metrics['density'])
        annealer.spec_optimize(metrics, graph)

        # Retrieve trajectories & calculate weights
        df = annealer.log__
        weights = {
            metric: 1/df[f"m_{metric}"].diff().abs().mean() for metric in metrics.keys()
        }

        # Update weights
        annealer.metrics_weights = weights
        
        # Transform ER graph
        annealer.optimize(graph)

        # Retrieve trajectories & calculate optimal beta
        df = annealer.log__
        beta = -log(0.23) * 1/df['loss'].diff().abs().mean()

        # Store beta and weights
        weights['beta'] = beta
        parameters.append(weights)

        print(df)
        print("Beta:", beta)
        print("Weights:", weights)

    parameters = pd.DataFrame(parameters)
    parameters.info()
    parameters.dropna(inplace=True)
    parameters = parameters.mean().to_dict()

    print("Measeured parameters:", parameters)

    # Save to yaml
    with open(params_file,'w') as file:
        yaml.dump(parameters, file, default_flow_style=False)
        
weights(snakemake.params.targets,
        snakemake.input[0],
        snakemake.output[0],
        snakemake.params[0]['alpha'],
        snakemake.params[0]['n_trials'],
        snakemake.params[0]['n_iterations'])
    