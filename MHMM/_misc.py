#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:35:16 2019

@author: george

miscelleneous help functions for MHMM Project


"""

import numpy as np
from scipy.special import logsumexp



def make_supervised( states_matrix, value = 0, value2 = None, value3 = 2):
    """
    takes a matrix with values 
    (in general 0 or 1) and produces
    a matrix with 1 and -infinities
    replacing the value "value" with -inf
    
    value2 : what value to replace with value 3
    """
    
    dim0 = states_matrix.shape[0]
    new_mat = np.zeros_like( states_matrix )
    
    for i in np.arange( dim0 ):
        
        rowi = states_matrix[i,:]
        rowi[np.where(rowi == value)] = -np.Inf
        
        if value2 is not None:
            rowi[np.where(rowi == value2)] = value3
            
        
        new_mat[i,:] = rowi
        
    
    return new_mat


def make_supervised2( states_matrix, drop = 0.7):
    """
     
    drop randomly a percentage of the state labels
    
   
    """
    
    perc = drop
    
    N, T = states_matrix.shape
    
    states_flat = states_matrix.reshape(N*T)
    
    #pick a random percentage of indexes to hide
    indx = np.random.choice( np.arange(N*T), size = int(perc*N*T), 
                            replace = False)
    
    
    
    states_flat[indx] = -np.inf
    
    new_mat = states_flat.reshape([N,T])
    
    return new_mat

def make_supervised3( states_matrix, drop = 0.8, drop_value = 1):
    
    N, T = states_matrix.shape
    
    st_flat = states_matrix.reshape(N*T)
    
    st_flat[ st_flat == 0] = -np.inf
    
    ind_val = np.where( st_flat == drop_value)[0]

    indx_drop = np.random.choice( len(ind_val), size = int(len(ind_val)*0.8))

    st_flat[indx_drop] = -np.inf
    
    print("Ones before: {} Ones after: {}".format(len(ind_val), len(np.where(st_flat == 1)[0])))
    return st_flat.reshape([N,T])
    


def dont_drop( values, states = None, drop_perc = 0):
    """
    make -infinity all the values in the states
    except the values "values"
    
    """    
    
    N,T = states.shape
    
    st = states.reshape(N*T)
    
    if drop_perc == 0:
        st[ ~np.isin(st, values) ] = -np.inf
        
    else:
        
        indx = np.random.choice( np.arange(N*T), size = int(drop_perc*N*T), 
                            replace = False)
        
        st[ indx ] =   -np.inf
    
    return st.reshape([N,T])
        
    

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





    