#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 22:46:37 2019

@author: george
"""
from sklearn.cluster import KMeans
import numpy as np
from _kmeans import custom_kmeans


def _kmeans_init(X, K, L, dates = None, states = None):
    """
    initializing with kmeans the HMM
    K = #states of HMM
    L: # of Gaussian Components of HMM
    
    """
    N = X.shape[0]
    T = X.shape[1]
    D = X.shape[2]
    
    #states = None
    
    covs = np.zeros( shape = [K, L, D, D] )
    means = np.zeros( shape = [K, L, D] ) 
    alphas = np.zeros( shape = [K,L] )
    
    if dates is None:
        X2d = X.reshape((N*T, D))
        if states is not None:
            states1d = states.reshape((N*T))
    
    else:
        
        X2d = _make3d_2d(X, dates)
        if states is not None:
            states1d = _make2d_1d(states, dates)
        
        
    #get the corresponding datasets for kmeans for each mixture
    if states is None: #if states are not given just do kmeans
        print('running hier')
        indexes = _hierarchical(X2d, K)
    
    else: #if states are given it is a bit more complicated
        print('running hier 3')

        indexes = _hierarchical3(X2d, states1d, K)
    
    
    
    for k in range(K): #for each state compute means labels, alphas
        ind_k = indexes[k]
       
        alphas[k], means[k], covs[k] = _compute_stats( X2d[ind_k], L)
        
    print(means)
    return alphas, means, covs
    
def _compute_stats(X, L):
    """
    calculates means, covariances, priors for
    L kmeans initialization
    """
    N = X.shape[0]
    D = X.shape[1]
    
    if L != 1:
        indexes = _hierarchical(X, L)
    else:
        indexes = [np.arange(len(X))]
    
    means = np.zeros( shape = [L,D] )
    covs = np.zeros( shape = [L,D,D] )
    alphas = np.zeros( shape = [L] )
    
    for l in  range(L):
        ind_l = indexes[l]
        means[l] = _compute_means(X[ind_l])
        covs[l] = _compute_covs(X[ind_l])
        alphas[l] = len(ind_l)/N
    
    return alphas, means, covs
    
def _hierarchical(X,  k, random_state = 0):
    """
    takes a NxD array and return the 
    indexes of Kmeans belonging to each of k classes
    
    """
    
    kmeans = KMeans(n_clusters = k, random_state = random_state)
    kmeans = kmeans.fit(X)
    labels = kmeans.labels_
    
    indexes = []
    for i in range(int(k)):
        indexes.append( np.where(labels == i)[0] )
        
    return indexes

def _hierarchical2(X, states1d, k, random_state = 0, value = 1):
    """
    takes a NxD array and return the 
    indexes of Kmeans belonging to each of k classes
    according to states
    now supports one label needs to be generalized
    
    """
    N = len(X)
    indexes = []
    #find unique values 
    #We assume that we have labels for all states or for the last state
    #the one with biggest unique number
    
    unique_values = np.unique( states1d )
    #check how many unique
    if np.isinf(unique_values[0]):
        k_kmeans = k-1
        
        if k_kmeans == 1:
            ind_inf = np.where( np.isinf(states1d))[0]
            ind_next = np.where( states1d == unique_values[1])[0]
            
            indexes = [ind_inf, ind_next]
    
        else: #do kmeans with k-1 classes
            
             #print("We are here", unique_values[-1])
             index_known = np.where(states1d == unique_values[-1])[0]
             #print(len(index_known),'Print me')
             ones = np.ones(shape = [N])
             ones[index_known] = 0
             kmeans = KMeans(n_clusters = k_kmeans, random_state = random_state)
             kmeans = kmeans.fit(X, sample_weight = ones)
             labels = kmeans.labels_
   
             for i in range( int(k_kmeans ) ):
                 
                 indx = np.setdiff1d(np.where(labels == i)[0], index_known)
                 indexes.append( indx )
             
             indexes.append( index_known )
    else: 
             #print('Here')
             #print(states1d.shape)

             for i in range( k ):
                 
                 indx = np.where( states1d == i)[0]
                 #print(len(indx))
                 indexes.append( indx )
                 
    
    #check length:
    Nind = 0
    for ind in indexes:
        #print( len(ind) )
        Nind += len(ind)
        
    #print(N, Nind)    
    return indexes


def _hierarchical3(X, states1d, k, random_state = 0, value = 1):
    
    """
    initialization with super clusters
    
    """
    N = len(X)
    unique_values = np.unique( states1d )
    
    if np.isinf(unique_values[0]):
        print('In h3 if')
        means, indexes, J, st_means, means_o = custom_kmeans(X, states1d, k)
        
        
    else: 
             #print('Here')
             #print(states1d.shape)
             print('In h3 else')

             indexes = []
             for i in range( k ):
                 
                 indx = np.where( states1d == i)[0]
                 #print(len(indx))
                 indexes.append( indx )

    #check length:
    Nind = 0
    for ind in indexes:
        #print( len(ind) )
        Nind += len(ind)
        
    print(N, Nind)    
    return indexes
    

def _make3d_2d(X, dates):
    
    """
    makes a 3d array 2d according to its dates
    """
    N = X.shape[0]
    D = X.shape[2]
    Nnew = int(np.sum(dates[:,1] - dates[:,0] + 1))
    
    X2d = np.zeros( shape = [Nnew, D])
    c = 0
    for i in range(N):
        #start end dates
        start = int(dates[i,0])
        end = int(dates[i,1])
        
        #number of time points
        s = end - start + 1
       # print("start:", start, " end:", end, " index",i)
        #print(c, s,  X[i, start : end + 1].shape, X2d[ c : c+s].shape)
        
        X2d[ c : c+s] = X[i, start : end + 1]
    
        #increasing the counter
        c += s
    

    return X2d

def _make2d_1d(states, dates):
    """
    makes a 2d array to 1d according to its states
    
    """
    
    N = states.shape[0]
   
    Nnew = int(np.sum(dates[:,1] - dates[:,0] + 1))
    
    states1d = np.zeros( Nnew)
    c = 0
    for i in range(N):
        #start end dates
        start = int(dates[i,0])
        end = int(dates[i,1])
        
        #number of time points
        s = end - start + 1
       # print("start:", start, " end:", end, " index",i)
        #print(c, s,  X[i, start : end + 1].shape, X2d[ c : c+s].shape)
        
        states1d[ c : c+s] = states[i, start : end + 1]
    
        #increasing the counter
        c += s
    
    
    return states1d
    
    
def _compute_means( X ):
     """
     computes mean vector
     
     """
     
     return np.mean( X, axis = 0 ) 
 

def _compute_covs( X ):
    
    """
    compute covarinace matrix
    """
    
    varX = np.var(X, axis = 0)
    dim = varX.ndim
    
    if dim > 0:
        diag = np.diag(varX + 1)
        
    else:
        diag = np.zeros(shape = [1,1])
        diag[0] = varX + 1
    
    return diag 
     
     
     
     
     
     
     
     
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    