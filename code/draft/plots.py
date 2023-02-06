# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:03:32 2021

@author: keira
"""

import numpy as np
import matplotlib.pyplot as plt

def distribution(x, y, xlabel = '', ylabel = '', bins = 50, legend = True, xmax = None):   
    # Define variables
    sig = (y == 1)
    common = dict(bins = bins, alpha = 0.5)

    # Get weights
    nsig = np.sum(sig)
    nback = np.sum(~sig)
    wsig = np.ones(nsig)/float(nsig)
    wback = np.ones(nback)/float(nback)
    
    # Plot distribution
    fig, ax = plt.subplots()
    ax.hist(x[sig], weights = wsig, color = 'b', label = 'Signal', **common)
    ax.hist(x[~sig], weights = wback, color = 'r', label = 'Background', **common)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or 'Fraction of events')
    ax.set_xlim(left = 0, right = xmax)
    if legend == True:
        ax.legend()
    
    return ax
    