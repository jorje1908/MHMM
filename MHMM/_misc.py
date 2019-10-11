#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:35:16 2019

@author: george
"""

import numpy as np



def make_supervised( states_matrix, value = 0):
    """
    takes a matrix with values 
    (in general 0 or 1) and produces
    a matrix with 1 and -infinities
    replacing the value "value" with -inf
    """
    
    dim0 = states_matrix.shape[0]
    new_mat = np.zeros_like( states_matrix )
    
    for i in np.arange( dim0 ):
        
        rowi = states_matrix[i,:]
        rowi[np.where(rowi == value)] = -np.Inf
        
        new_mat[i,:] = rowi
        
    
    return new_mat
        
    

def compute_forw(hmm, data):
    """
 
    computes the forward probabilities for all data
    
    """
    
    N = data.shape[0]
    T = data.shape[1]
    
    zers = np.zeros(shape = [N,T])
    ones = np.zeros( shape = [N,T])
    
    for  i in range(N):
        forw = np.exp( hmm.log_forward(data[i,:,:]) )
        zers[i] = forw[0,:]
        ones[i] = forw[1,:]
        
    return zers.reshape(N*T), ones.reshape(N*T)
    