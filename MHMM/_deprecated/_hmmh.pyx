#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:55:00 2019

@author: george
"""

#cython: language_level=3, boundscheck=False, wraparound=False
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from cython cimport view

#cimport numpy as np

import numpy as np
from numpy import logaddexp
from scipy.special import logsumexp

ctypedef double dtype_t

#FUNCTIONS FROM hmmlearn Project
cdef inline int _argmax(dtype_t[:] X) :
    """
    finds the maximum logarithm in matrix
    X
    returns the position of max
    to be used by the _max function
    """
    cdef dtype_t X_max = -np.math.inf
    cdef int pos = 0
    cdef int i
    
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
            
    return pos


cdef inline dtype_t _max(dtype_t[:] X) :
    """
    returns max element of X
    """
    return X[_argmax(X)]


cdef inline dtype_t _logsumexp(dtype_t[:] X) :
    """
    takes the exponential of all the elements in X
    adds them after subtracting the max X for stability
    and then takes the logarithm again
    """
    
    cdef dtype_t X_max = _max(X)
    
    if np.isinf(X_max):
        return -np.math.inf

    cdef dtype_t acc = 0
    cdef int i
    cdef int N = X.shape[0]
    
    for i in range(N):
        acc += np.math.exp(X[i] - X_max)

    return np.log(acc) + X_max


cdef inline dtype_t _logaddexp(dtype_t a, dtype_t b) :
    
    if np.isinf(a) and a < 0:
        return b
    elif np.isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + np.math.log(1+np.math.exp(-np.math.fabs(a - b)))





def _forward( dtype_t[:,:] A,  dtype_t[:,:] p_states,
              dtype_t[:] init_states,  dtype_t[:,:] forw,
              int T,  int K):
    
    """
    
    Cython implementation of forward algorithm
    
    """
    
    cdef int t,i,j 
    
    with nogil:
    
        for i in range(K):
            forw[i,0] = init_states[i]*p_states[i,0]
    
        for t in range(1, T):
                for i in range(K):
                    for j in range(K):
                        forw[i,t] +=  A[j,i]*forw[j,t-1]
                   
                    forw[i,t] *= p_states[i,t]
                    
                    
                    
def _backward( dtype_t[:,:] A,  dtype_t[:,:] p_states,
                   dtype_t[:] init_states,  dtype_t[:,:] backw,
                                              int T,  int K):
    """
    Cython Implementation of backward
    """
    
    cdef int t,i,j 
    
    with nogil:
        
        for i in range(K):
            backw[i,T-1] = 1
        
        
        for t in range( T-2, -1, -1):
            for i in range(K):
                for j in range(K):
                    backw[i,t] += A[i,j]*backw[j,t+1]*p_states[j,t+1]
                    
def _xis_log( dtype_t[:,:] log_A,  dtype_t[:,:] log_p_states,
                     dtype_t[:,:] log_forw,
                   dtype_t[:,:] log_backw, dtype_t[:,:,:] log_xis, int T, int K):
    
    """
    calculates sliced probabilities
    """
    
    cdef int i,j,t
    cdef dtype_t[:, ::view.contiguous] work_buffer = \
        np.full((K, K), -np.math.inf)
    
    cdef dtype_t logzero
        
    
    
    for t in range(T-1):
        logzero = -np.math.inf
        for i in range(K):
            for j in range(K):
                log_xis[i,j,t] = log_forw[i, t] + log_backw[j, t+1]\
                                  +log_A[i,j] + log_p_states[j,t+1] 
                
                logzero = logaddexp(logzero, log_xis[i,j,t])
                
        for i in range(K):
            for j in range(K):
                log_xis[i,j,t] = log_xis[i,j,t] - logzero
                                 
        
                                       
            
                    
                    
def _forward_log( dtype_t[:,:] log_A,  dtype_t[:,:] log_p_states,
                   dtype_t[:] log_init_states,  dtype_t[:,:] log_forw,
                                              int T,  int K):
    
    """
    Cython implementation of log forward
    
    """
    
    
    cdef int t,i,j 
    cdef dtype_t[::view.contiguous] work_buffer = np.zeros(K)
    
   
        
    for i in range(K):
        log_forw[i,0] = log_init_states[i] + log_p_states[i,0]
            
            
    for t in range(1,T):
        for i in range(K):
            for j in range(K):
                work_buffer[j] = log_A[j,i] + log_forw[j, t-1]
                    
            log_forw[i, t] = logsumexp(work_buffer) + log_p_states[i,t]
                    
                    
                    
def _backward_log( dtype_t[:,:] log_A,  dtype_t[:,:] log_p_states,
                   dtype_t[:,:] log_backw, int T,  int K):
    
    """
    Cython implementation of log backward
    """
    
    cdef int t,i,j 
    cdef dtype_t[::view.contiguous] work_buffer = np.zeros(K)
    
   
        
    for i in range(K):
        log_backw[i,T-1] = 0
        
        
    for t in range( T-2, -1, -1):
        for i in range(K):
            for j in range(K):
                work_buffer[j] += log_A[i,j]+log_backw[j,t+1]+log_p_states[j,t+1]
                
            log_backw[i, t] = logsumexp(work_buffer)
                
                
def _sum_xi_log( dtype_t[:,:] log_A,  dtype_t[:,:] log_p_states,
                   dtype_t[:] log_init_states,  dtype_t[:,:] log_forw,
                   dtype_t[:,:] log_backw, dtype_t[:,:] log_xi_sum, int T,  int K):
    """
    sum of the xis
    for the EM 
    
    """
    
    cdef int i,j,t
    cdef dtype_t[:, ::view.contiguous] work_buffer = \
        np.full((K, K), -np.math.inf)
        
    cdef dtype_t logprob = _logsumexp(log_forw[:,T - 1])
    
    
    
    for t in range(T):
        for i in range(K):
            for j in range(K):
                work_buffer[i,j] = log_forw[i, t] + log_backw[j, t+1]\
                                       +log_A[i,j] + log_p_states[j,t+1] \
                                       -logprob
                                       
        for i in range(T):
            for j in range(K):
                log_xi_sum[i,j] = _logaddexp(log_xi_sum[i, j], work_buffer[i, j])
                
    
    
    
    
                        
                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    