#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:57:36 2019

@author: george
"""

import numpy as np
from numpy import logaddexp
#from scipy.special import logsumexp


def logsumexp(a, axis = 0):
    
    max_a = np.max(a, axis = axis)
    exp_a = a - max_a
    np.exp(exp_a, out=exp_a)
    
    c = np.log(np.sum(exp_a, axis = axis))
   
    c += max_a 
    return c

def _log_forward( log_A, log_p_states, log_init_states, log_forw, T, K,
                 states = None):
    
    """
    implements forward algorithm
    log_A: log transition matrix
    log_p_states: KxT matrix probability of each point in time
                  given the state
    log_init_states = initial state probability dist
    
    log_forw: matrix to get the results
    T: #of time steps
    K: #states
    states: states labels if available
    """
    
    
        
    log_forw[:,0] = log_p_states[:,0] + log_init_states
        
        
    if states is not None:
        if not np.isinf(states[0]):
            s0 = int(states[0])
            
            helpMat = logsumexp(log_forw[:,0])
            log_forw[:,0] = -np.inf
            log_forw[s0, 0] = helpMat.copy()
            
    #added           
    N0 = logsumexp(log_forw[:,0])
    log_forw[:,0] -=N0   
    Ntsum = N0
    ######
    
    work_buffer  = np.zeros(shape = [K,K])
    
    for t in range(1,T):
        N = -np.inf
        
        work_buffer = log_A + log_forw[:,t-1].reshape(-1,1)
            
        log_forw[:,t] = logsumexp(work_buffer, axis = 0) + log_p_states[:,t]
        N = logaddexp( logsumexp(log_forw[:,t]), N ) 
        Ntsum = logaddexp(N, Ntsum)
        log_forw[:,t] -= N
        
        if states is not None:
            if not np.isinf( states[t]):
                st = int(states[t])
                helpMat = logsumexp(log_forw[:,t])
                log_forw[:,t] = -np.inf
                log_forw[st,t] = helpMat.copy()
        
    return Ntsum
                

def _log_viterbi(log_A, log_p_states, log_init_states, log_vit, T, K):
    
    states = np.zeros_like( log_vit )
    states[:,0] = np.arange(K)
    
    for i in range(K): #initialize
        log_vit[i,0] = log_p_states[i,0] + log_init_states[i]
    
    N0 = logsumexp(log_vit[:,0])
    log_vit[:,0] -= N0
    work_buffer  = np.zeros(shape = [K])  
    
    for t in range(1,T):
        N = -np.inf
        for i in range(K):
            for j in range(K):
                work_buffer[j] = log_A[j,i] + log_vit[j,t-1]
            
            N = logaddexp(logsumexp(work_buffer)+ log_p_states[i,t], N)
            log_vit[i,t] = max(work_buffer) + log_p_states[i,t]
            states[i,t] = np.argmax(work_buffer)
        #normalize
        log_vit[:,t] -= N
        
    max_seq = []
    max_prob_ind = int(np.argmax(log_vit[:,T-1]))
    max_seq.append(max_prob_ind)
    
    for t in range(T-1, 0, -1):     
        max_seq.append(states[max_prob_ind, t])
        max_prob_ind = int(states[max_prob_ind,t])
        
    max_seq.reverse()  
    return log_vit, states, max_seq
                
            
          
            
def _log_backward(log_A, log_p_states, log_backw, T, K):
    
    
    
    log_backw[:,T-1] = 0
        
    
    work_buffer  = np.zeros(shape = [K,K])
    for t in range(T-2, -1, -1):
        work_buffer = log_A + log_backw[:,t+1] + log_p_states[:,t+1]
            
        log_backw[:,t] = logsumexp(work_buffer, axis = 1)  
            
            
def _log_gamas(log_forw, log_backw, log_gammas):
    
    log_gammas = log_forw + log_backw
    
    normalize = logsumexp(log_gammas, axis = 0)
    log_gammas = log_gammas - normalize
    
    return log_gammas
    
    
def _log_xis(log_A, log_p_states, log_forw, log_backw, log_xis, T, K):
    
    
    for t in range(T-1):
        logzero = -np.math.inf
        log_xis[:,:,t] = (log_A +log_forw[:, t].reshape(-1,1)) + (log_backw[:, t+1]
                                   + log_p_states[:,t+1] )
                
        logzero = logaddexp(logzero, logsumexp(log_xis[:,:,t]))
        log_xis[:,:,t] = log_xis[:,:,t] - logzero
    
    
    
        


           
        
        
        
        








        
    