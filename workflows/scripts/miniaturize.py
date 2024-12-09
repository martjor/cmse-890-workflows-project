#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@msu.edu
last updated April 2024

Parallel tempering definition

REQUIRES: Metropolis.py

to run:

mpiexec -n $number_of_nodes python parallel_tempering_main.py
'''
import click
from mpi4py import MPI
import numpy as np
from minigraphs.miniaturize import MH
import networkx as nx
from scipy.sparse import save_npz
import os
import sys
import yaml
import pandas as pd
from utils import StreamToLogger
import logging

@click.command()
@click.argument('metrics-file',type=click.Path(exists=True))
@click.argument('params-file',type=click.Path(exists=True))
@click.argument('adjacency-file',type=click.Path())
@click.argument('trajectories-dir',type=click.Path())
@click.argument('shrinkage',type=click.FLOAT)
@click.option('--n_changes',default=10,help='Number of changes proposed at each iteration')
@click.option('--n_steps',default=20000,help='Number of miniaturization steps')
@click.option('--n_substeps',default=200,help='Number of miniaturization substeps')
@click.option('--log-file',default=None)
def miniaturize(metrics_file,
                params_file,
                adjacency_file,
                trajectories_dir,
                shrinkage,
                n_changes,
                n_steps,
                n_substeps,
                log_file):
    
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
    
    ##### INITIALIZE MPI #####
    #========================#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.size
    
    ##### READ INPUT FILES #####
    #==========================#
    if size % 2:
        raise Exception('An even number of cores is required')

    # Target metrics
    with open(metrics_file,'r') as file:
        metrics = yaml.safe_load(file)
    
    # Miniaturization Parameters
    with open(params_file,'r') as file:
        params = yaml.safe_load(file)

    ##### INITIALIZE PARALLEL TEMPERING REPLICAS #####
    #================================================#
    # Iteration parameters
    cycles = [n_substeps] * (n_steps // n_substeps)
    remainder = n_steps % n_substeps
    if remainder != 0:
        cycles += [remainder]

    # Replica parameters
    n_vertices = int((1-shrinkage) * metrics['n_vertices'])
    beta_arr = np.array([1/8,1/4,1/2,1,2,4])
    B0 = beta_arr[rank] * params['beta']
    metrics_funcs = {
        'density': nx.density,
        'assortativity_norm': lambda G: (nx.degree_assortativity_coefficient(G)+1)/2,
        'clustering': nx.average_clustering
    }

    metrics_target = {
        key:metrics[key] for key in metrics_funcs.keys()
    }

    weights = {
        key:params[key] for key in metrics_funcs.keys()
    }
        
    # Initialize seed graph
    G = nx.erdos_renyi_graph(n_vertices,metrics_target['density'])

    # Display Message
    if rank == 0:
        print(f"Miniaturizing graph to size {n_vertices} ({shrinkage * 100}% miniaturization)...")
        print(f"Beta opt: {params['beta']}\n")
        print(f"\t - Target metrics: {metrics_target}")
        print(f"\t - Weights: {weights}")
        print(f"\t - Functions: {metrics_funcs}\n")

    # Initialize replica
    replica = MH(metrics_funcs,
                 schedule=lambda beta:0,
                 n_changes=n_changes,
                 weights=weights)

    def exchange(E0: float,
                 B0: float) -> float:
        '''Replicas exchange temperatures according to their energies
        '''
        def prob(E0,E1,B0,B1) -> float:
            '''Defines exchange probability according to the metropolis criterion
            '''
            return min(1.0,np.exp((E0-E1)*(B0-B1)))
        
        def swap(E0,B0,flag_send,flag_recv,ref) -> float:
            '''Even ranks propose energy change in the specified direction
            '''
            if flag_send[rank]:
                # Send energy and temp forward to next rank
                comm.send(E0, dest=rank+ref, tag=0)
                comm.send(B0, dest=rank+ref, tag=1)
                
                # Receive new temp
                B0 = comm.recv(source=rank+ref, tag=2)

            elif flag_recv[rank]:
                # Receive energy and temp from previous rank
                E1 = comm.recv(source=rank-ref, tag=0)
                B1 = comm.recv(source=rank-ref, tag=1)
                
                # Accept or reject switch
                if prob(E0,E1,B0,B1) > np.random.uniform(0,1):
                    comm.send(B0, dest=rank-ref, tag=2)
                    B0 = B1
                    
                else: 
                    comm.send(B1, dest=rank-ref, tag=2)
                    
            # Sync replicas
            comm.Barrier()
            
            return B0
            
        # Get indices of senders
        flag_recv = (np.arange(0,len(beta_arr)) % 2).astype(bool)
        flag_send = np.invert(flag_recv)
        
        # Send forward
        B0 = swap(E0,B0,flag_send, flag_recv, ref=1)
        
        # Send backward
        flag_send[0] = False
        flag_recv[-1] = False
        B0 = swap(E0,B0,flag_send, flag_recv, ref=-1)
        
        return B0

    list_trajectories = []
    for cycle, steps in enumerate(cycles):
        if rank == 0:
            print(f"\nCycle {cycle+1}/{len(cycles)} <<<\n")
            verbose = True 
        else:
            verbose = False
            
        # Optimize graph
        replica.transform(G,
                          metrics_target,
                          n_iterations=steps)
        
        # Swap temperatures
        trajectories = replica.trajectories_.copy()
        E0 = trajectories['Energy'].iat[-1]
        B0 = replica.beta

        # Update schedule
        replica.schedule = lambda beta: exchange(E0,B0)
        
        G = replica.graph_
        
        # Store trajectories
        list_trajectories.append(replica.trajectories_.copy())

    # Create complete trajectories
    trajectories_all = pd.concat(list_trajectories)

    # Store your energy and rank
    my_energy_core = np.array([(E0,rank)],dtype=[('energy',np.float64), ('rank',np.int32)])
    # allocate memory for min energy and rank
    min_energy_core = np.empty(1,dtype=[('energy',np.float64), ('rank',np.int32)])

    # reduce to all ranks the mimum energy and rank that has the mimimum
    comm.Allreduce([my_energy_core, 1, MPI.DOUBLE_INT], [min_energy_core, 1, MPI.DOUBLE_INT], op=MPI.MINLOC)

    if rank == min_energy_core['rank']:
        # Store final adjacency
        save_npz(adjacency_file,nx.to_scipy_sparse_array(replica.graph_))
        
        # Create directory for trajectories
        os.makedirs(trajectories_dir)
        
    # Store trajectories
    comm.Barrier()
    trajectories_all.to_parquet(f"{trajectories_dir}/replica_{rank}.parquet",
                                engine='pyarrow',
                                compression='snappy')
    
if __name__ == '__main__':
    miniaturize()


