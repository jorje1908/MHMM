#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 22:46:37 2019

@author: george
"""
from sklearn.cluster import KMeans
import numpy as np


def _kmeans_init(X, K, L, dates = None):
    """
    initializing with kmeans the HMM
    K = #states of HMM
    L: # of Gaussian Components of HMM
    
    """
    N = X.shape[0]
    T = X.shape[1]
    D = X.shape[2]
    
    covs = np.zeros( shape = [K, L, D, D] )
    means = np.zeros( shape = [K, L, D] ) 
    alphas = np.zeros( shape = [K,L] )
    
    if dates is None:
        X2d = X.reshape((N*T, D))
    
    else:
        
        X2d = _make3d_2d(X, dates)
        
        
    #get the corresponding datasets for kmeans for each mixture
    indexes = _hierarchical(X2d, K)
    
    for k in range(K): #for each state compute means labels, alphas
        ind_k = indexes[k]
        alphas[k], means[k], covs[k] = _compute_stats(X2d[ind_k], L)
        
    
    return alphas, means, covs
    
def _compute_stats(X, L):
    """
    calculates means, covariances, priors for
    L kmeans initialization
    """
    N = X.shape[0]
    D = X.shape[1]
    indexes = _hierarchical(X, L)
    
    means = np.zeros( shape = [L,D] )
    covs = np.zeros( shape = [L,D,D] )
    alphas = np.zeros( shape = [L] )
    
    for l in  range(L):
        ind_l = indexes[l]
        means[l] = _compute_means(ind_l)
        covs[l] = _compute_covs(ind_l)
        alphas[l] = len(ind_l)/N
    
def _hierarchical(X, k, random_state = 0):
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


def _make3d_2d(X, dates):
    
    """
    makes a 3d array 2d according to its dates
    """
    N = X.shape[0]
    D = X.shape[2]
    Nnew = np.sum(dates[1] - dates[0] + 1)
    
    X2d = np.zeros( shape = [Nnew, D])
    c = 0
    for i in range(N):
        #start end dates
        start = dates[i,0]
        end = dates[i,1]
        
        #number of time points
        s = end - start + 1
        
        
        X2d[ c : c + s, :] = X[i, start : end + 1, :]
    
        #increasing the counter
        c += s

    return X2d
    
    
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
    return np.diag(varX)
     
     
     
     
     
     
     
     
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    