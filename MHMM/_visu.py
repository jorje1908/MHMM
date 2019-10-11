#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:33:10 2019

@author: george
"""

import numpy as np
import matplotlib.pyplot as plt



def plot_prob_state( state_probs, state = 1, index_real = None, 
                                            real_state = None, name = None):
    """
    plots the probability of being at given state
    at each time step
    
    if there is a real state it gives the real state index
    so you can know that this is where there is the real state
    """
    
   
        
        
    state_index = np.arange(state_probs.shape[0])
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(state_index, state_probs)
    try:
        ax.plot(index_real, real_state )
        ax.legend(["{} Probability of being at state {}".format(name,state), "REAL STATE"])
    except:
        ax.legend(["{} Probability of being at state {}".format(name,state)])

    ax.set_xlabel("STATES")
    ax.set_ylabel("PROBABILITIES")
    
    return
    
    