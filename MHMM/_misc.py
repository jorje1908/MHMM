#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:35:16 2019

@author: george

miscelleneous help functions for MHMM Project


"""

import numpy as np
from scipy.special import logsumexp



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
    
    gets an hmm and the data
    
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


#########     check functions   #########
def checkShape(  arg1, arg2, name):
        
    if arg1.shape != arg2.shape:
        print( "Warning shapes does not match in " + name)
        
        
    return

def checkSum_one( matrix, axis, name):
    """
    Checks if the matrix entries along the given axis
    sum to 1
    """
    
    result = matrix.sum( axis = axis ).round(5)
    value = np.all( result == 1 )
    
    
    if not value:
        print(" Warning: Elements do not sum to 1 in {} ".format(name))
        
    return

def checkSum_zero( matrix, axis, name):
    """
    Checks if the matrix entries along the given axis
    sum to 0
    """
    
    result = logsumexp(matrix, axis = axis ).round(5)
    value = np.all( result == 0 )
    
    
    if not value:
        print(" Warning: Elements do not sum to 0 in {} ".format(name))
        
    return


def make_dataset(X, points):
        """
        helper function for the Kmeans Initialization
        
        returns a dataset with points number of observations from X
        """
        T = X[0].shape[0]
        N = len( X )
        d = X[0].shape[1] 
        
        #see how many points we need to concatenate together
        indx_num = int( np.ceil( points/ T ) )
        #choose random indexes
        indx = np.random.choice( np.arange(N), size = indx_num, replace = False)
        indx = indx.astype(int)
        
        #return the Kmeans dataset
        X_kmeans = X[indx]
        X_kmeans = np.reshape( X_kmeans, [-1, d])
        
        return X_kmeans





    