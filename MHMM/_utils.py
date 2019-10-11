#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:03:48 2019

@author: george
"""

import numpy as np
from numpy import logaddexp
from scipy.special import logsumexp




def _log_forward( log_A, log_p_states, log_init_states, log_forw, T, K,
                 states = None):
    
    
    for i in range(K):#initialize
        
        log_forw[i,0] = log_p_states[i,0] + log_init_states[i]
        
        
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
    
    work_buffer  = np.zeros(shape = [K])
    for t in range(1,T):
        N = -np.inf
        for i in range(K):
            for j in range(K):
                work_buffer[j] = log_A[j,i] + log_forw[j,t-1]
            
            log_forw[i,t] = logsumexp(work_buffer) + log_p_states[i,t]
            N = logaddexp(log_forw[i,t], N)
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
    
    
    for i in range(K):
        log_backw[i,T-1] = 0
        
    
    work_buffer  = np.zeros(shape = [K])
    for t in range(T-2, -1, -1):
        for i in range(K):
            for j in range(K):
                work_buffer[j] = log_A[i,j] + log_backw[j,t+1] + \
                                                            log_p_states[j,t+1]
            
            log_backw[i,t] = logsumexp(work_buffer)  
            
            
def _log_gamas(log_forw, log_backw, log_gammas):
    
    log_gammas = log_forw + log_backw
    
    normalize = logsumexp(log_gammas, axis = 0)
    log_gammas = log_gammas - normalize
    
    return log_gammas
    
    
def _log_xis(log_A, log_p_states, log_forw, log_backw, log_xis, T, K):
    

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
    
    
    
        


           
        
        
        
        








        
    