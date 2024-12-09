#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@lanl.gov

The following code plots the results from degroot.py

to run:

python degroot.py $input_directory
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
    plt.plot(data)

plt.show()
