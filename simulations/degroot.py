#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@lanl.gov

The following code simulates a the degroot model

to run:

python degroot.py $path_to_graph.npz
'''

import numpy as np
import os
import sys
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
import datetime

#################
# Gloabal Setup #
#################

# name of graph that code will run on
GRAPH_NAME = sys.argv[1]
# maximum allowed iterations
MAX_ITERS = int(sys.argv[2])
# name of output directory to save results
OUTPUT_DIR = sys.argv[3]



# make directory if it hasn't been made yet
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR,exist_ok=True)

# load the adjacency matrix for graph that the model will run on
# it is assumed that the graph will be an scipy.sparse matrix
A = load_npz(GRAPH_NAME)
T = normalize(A, axis=1, norm='l1')
# find the number of agents in the graph
N = A.shape[0]


x0 = np.random.uniform(0,1,size=N)
print(f"Running DeGroot on {GRAPH_NAME}")
print(f"Size {N}")

# plot_data = np.zeros((101,100))
trajectory = np.zeros((MAX_ITERS+1,N))
trajectory[0] = np.copy(x0)

# plot_data[0] = test_x

for i in range(1,MAX_ITERS+1):
    trajectory[i] = T@trajectory[i-1]
    # plot_data[i] = test_x

np.save(OUTPUT_DIR+'/run_'+datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S_%f'),trajectory)
