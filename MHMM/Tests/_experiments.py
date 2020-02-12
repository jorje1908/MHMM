#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:24:13 2019

@author: george
"""
import numpy as np
import scipy as sc

def generate_Coin(A = None, init = None, p_Coin = None, c_type = "Standard",
                  N = 5000):
    
     """
            coin experiment for HMM testing
     """
     if c_type == "Standard":
        A = np.zeros( shape = [2,2])
        A[0,0] = 0.5
        A[0,1] = 0.5
        A[1,0] = 0.05
        A[1,1] = 0.95
        
        init = np.zeros(2)
        init[0] = 0.5
        init[1] = 0.5
        
        p_Coin = np.zeros( shape = [2,2])
        p_Coin[0,0] = 0.99
        p_Coin[0,1] = 0.01
        p_Coin[1,0] = 0.01
        p_Coin[1,1] = 0.99
        
        
     states = np.zeros(shape = [N,1])
     coins = np.zeros( shape = [N,1])
     si = -1
     for i in range(N):
         if i == 0:
             si = np.random.choice([0,1], size = 1, p = init )[0]
        
         else:
             si = np.random.choice([0,1], size = 1, p = A[si])[0]
        
         states[i] = si
         ci = np.random.choice([0,1], size = 1,p = p_Coin[si])[0]
         coins[i] = ci


     data = np.expand_dims(np.concatenate((states.T, coins.T), axis = 0 ), axis = 2)
     return data, states, coins
 
    
def gauss_seq(T = 24, N = 500, A = None, mean = [0,1], 
                             std = [1,1]):
    
    """
    Generate a training set of N sequences of length T, to train
    Hidden Markov Model
    
    """
    if A is None:
        print(" NO A MATRIX INPUTED ")
        return
    
    data = np.zeros(shape = [N, T, mean.shape[1]] ) #dataset shape
    states = np.zeros(shape = [N, T] )
    
    for i in range(N):
        data[i,:,:], states[i] = generate_gauss(T, A, mean, std)
        
    return data, states
    
    
def generate_gauss( T, A, mean, std): 
    
    """
    
    generates a sequence of data points from an HMM
    with state transition model A and observational model
    1 gaussian in each state with means and variances m_0, m_1
    var_0, var_1
    
    """
    K = A.shape[0]
    sequence = np.zeros([T, mean.shape[1]])
    states = np.zeros(T)
    
    
    
    #start by picking a state at random
    s_t = np.random.choice(np.arange(K), size = 1)[0].astype(int)
    x_t = np.random.normal(loc = mean[s_t], scale = std[s_t], size = mean.shape[1])[0]
    
    
    states[0] = s_t
    sequence[0] = x_t
    
    for t in range(1, T):
        s_t = np.random.choice(np.arange(K), size = 1, p = A[s_t])[0].astype(int)
        x_t = np.random.normal(loc = mean[s_t], scale = std[s_t], size = mean.shape[1])
        
        states[t] = s_t
        sequence[t] = x_t
        
        
    return sequence, states

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    