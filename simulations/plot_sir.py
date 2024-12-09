#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@lanl.gov

The following code plots the results from SIR.py

to run:

python SIR.py $input_directory
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

# file to plot
INPUT_DIR = sys.argv[1]
plt.figure(figsize=(8,5))
for file in glob.glob(INPUT_DIR+"/*"):
    data=np.load(file)
    plt.plot(data[:,0],c='C0')
    plt.plot(data[:,1],c='C1')
    plt.plot(data[:,2],c='C2')

plt.plot(data[:,0],label='S',c='C0')
plt.plot(data[:,1],label='I',c='C1')
plt.plot(data[:,2],label='R',c='C2')
plt.legend()
plt.show()
