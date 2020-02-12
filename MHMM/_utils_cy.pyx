#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:58:24 2019

@author: george
"""



# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

cimport cython
from cython.parallel cimport prange
from libc.math cimport log, exp, INFINITY, isinf, fabs, log1p, fmax
import numpy as np
from numpy import logaddexp
from scipy.special import logsumexp
from libc.stdio cimport printf

ctypedef  double dt

#TAKEN FROM HMM LEARN PACKAGE
#argmax function
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline int _argmax(dt[:] X) nogil:
    cdef dt X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos

#return max of an array
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline dt _max(dt[:] X) nogil:
    return X[_argmax(X)]

#custom log sum exp
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline dt _logsumexp(dt[:] X) nogil:
    cdef dt X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dt acc = 0
    for i in range(X.shape[0]):
        acc += exp(X[i] - X_max)

    return log(acc) + X_max

#custom log add exp
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef inline dt _logaddexp(dt a, dt b) nogil:
    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return fmax(a, b) + log1p(exp(-fabs(a - b)))
    




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef dt _log_forward( dt[:,:] log_A, dt[:,:] log_p_states,  dt[:] log_init_states, 
                       dt[:,:] log_forw, int T, int K, dt[:] states, int flag):
    
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
    cdef int i 
    cdef int t 
    cdef int j,h
    cdef int s0,st
    cdef dt N0, Ntsum, N
    cdef dt[:] work_buffer = np.zeros(shape = [K])
    cdef dt helpMat
    
    for i in prange(K, nogil = True):#initialize
        
        log_forw[i,0] = log_p_states[i,0] + log_init_states[i]
        
    with nogil:   
        if flag == 1:
            if not isinf(states[0]):
                s0 = <int>(states[0])
                
                helpMat = _logsumexp(log_forw[:,0])
                log_forw[:,0] = -INFINITY
                log_forw[s0, 0] = helpMat
        #added           
        N0 = _logsumexp(log_forw[:,0])
        #printf('%f', N0)
        for h in prange(K):
            log_forw[h,0] =log_forw[h,0] - N0   
        #log_forw[:,0] = np.subtract(log_forw[:,0], N0)   
        Ntsum = N0
       # printf('%f', Ntsum)
        ######
        
        
        
        for t in range(1,T):
            N = -INFINITY
            for i in range(K):
                for j in prange(K):
                    work_buffer[j] = log_A[j,i] + log_forw[j,t-1]
                
                log_forw[i,t] = _logsumexp(work_buffer) + log_p_states[i,t]
                N = _logaddexp(log_forw[i,t], N)
                
            #Ntsum = _logaddexp(N, Ntsum)
            Ntsum = Ntsum + N
            #log_forw[:,t] = np.subtract(log_forw[:,t], N)  
            for h in prange(K):
                log_forw[h,t] = log_forw[h,t] - N
                
            if flag == 1:
                if not isinf( states[t] ):
                    st = <int>(states[t])
                    helpMat = _logsumexp(log_forw[:,t])
                    log_forw[:,t] = -INFINITY
                    log_forw[st,t] = helpMat
            #printf('Ntsum In for: %f', Ntsum)
    #printf('Ntsum last %f', Ntsum)   
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
                
            
          
 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.           
cpdef _log_backward(dt[:,:]log_A, dt[:,:] log_p_states, dt[:,:] log_backw, int T, int K):
    
    cdef int i
    cdef int t
    cdef int j
    cdef dt[:] work_buffer = np.zeros(shape = [K])
    
    with nogil:
        for i in prange(K):
            log_backw[i,T-1] = 0
            
        
        for t in range(T-2, -1, -1):
            for i in range(K):
                for j in prange(K):
                    work_buffer[j] = log_A[i,j] + log_backw[j,t+1] + \
                                                        log_p_states[j,t+1]
                
                log_backw[i,t] = _logsumexp(work_buffer)  
            
            
#def _log_gamas(log_forw, log_backw):
#    
#    log_gammas = log_forw + log_backw
#    
#    normalize = logsumexp(log_gammas, axis = 0)
#    log_gammas = log_gammas - normalize
#    
#    return log_gammas

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.  
cpdef dt[:,:] _log_gamas(dt[:,:] log_forw, dt[:,:] log_backw):
    
    cdef int K = log_forw.shape[0]
    cdef int T = log_forw.shape[1]
    cdef int i, k
    cdef dt[:,:] log_gammas = np.zeros( shape = [K,T] )
    cdef double normalize
    
    with nogil:
        for i in prange(T):
            for k in prange(K):
                log_gammas[k,i] = log_forw[k,i] + log_backw[k,i]
                
        for i in range(T):
            normalize = _logsumexp(log_gammas[:,i])
            for k in prange(K):
                log_gammas[k,i] = log_gammas[k,i] - normalize
            
    return log_gammas
    
    

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.   
def _log_xis(dt[:,:] log_A, dt[:,:] log_p_states, dt[:,:] log_forw, 
             dt[:,:] log_backw, dt[:,:,:] log_xis, int T, int K):
    
    cdef int t,i,j
    cdef dt logzero
    
    with nogil:
        for t in range(T-1):
            logzero = -INFINITY
            for i in range(K):
                for j in range(K):
                    log_xis[i,j,t] = log_forw[i, t] + log_backw[j, t+1]\
                                      +log_A[i,j] + log_p_states[j,t+1] 
                    
                    logzero = _logaddexp(logzero, log_xis[i,j,t])
                    
            for i in prange(K):
                for j in prange(K):
                    log_xis[i,j,t] = log_xis[i,j,t] - logzero
                    
                    
                    
                    
#ADDING PROBABILITIES IN THE LABELS                  
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef dt _log_forward2( dt[:,:] log_A, dt[:,:] log_lab,
                       dt[:,:] log_p_states,  dt[:] log_init_states, 
                       dt[:,:] log_forw, int T, int K, dt[:] states, int flag):
    
    """
    implements forward algorithm
    log_A: log transition matrix
    log_lab: labels conditional probabilities p(label = i | z = k)
    log_p_states: KxT matrix probability of each point in time
                  given the state
    log_init_states = initial state probability dist
    
    log_forw: matrix to get the results
    T: #of time steps
    K: #states
    states: states labels if available
    """
    cdef int i 
    cdef int t 
    cdef int j,h, ii
    cdef int s0,st
    cdef dt N0, Ntsum, N
    cdef dt[:] work_buffer = np.zeros(shape = [K])
    cdef dt helpMat
    
    if  isinf( states[0] ):
            ii = 0
    else:
            ii = <int> (states[0]) + 1
   
    for i in prange(K, nogil = True):#initialize
        
        log_forw[i,0] = log_p_states[i,0] + log_init_states[i] + log_lab[i, ii]
        
    with nogil:   
#         if flag == 1:
#             if not isinf(states[0]):
#                 s0 = <int>(states[0])
                
#                 helpMat = _logsumexp(log_forw[:,0])
#                 log_forw[:,0] = -INFINITY
#                 log_forw[s0, 0] = helpMat
        #added           
        N0 = _logsumexp(log_forw[:,0])
        for h in prange(K):
            log_forw[h,0] =log_forw[h,0] - N0   
        #log_forw[:,0] = np.subtract(log_forw[:,0], N0)   
        Ntsum = N0
        ######
        
        
        
        for t in range(1,T):
            N = -INFINITY
            if  isinf( states[t] ):
                ii = 0
            else:
                ii = <int> (states[t]) + 1
                
            for i in range(K):
                for j in prange(K):
                    work_buffer[j] = log_A[j,i] + log_forw[j,t-1]
                
                log_forw[i,t] = _logsumexp(work_buffer) + log_p_states[i,t] + log_lab[i, ii]
                N = _logaddexp(log_forw[i,t], N)
                
            Ntsum = _logaddexp(N, Ntsum)
            #log_forw[:,t] = np.subtract(log_forw[:,t], N)  
            for h in prange(K):
                log_forw[h,t] = log_forw[h,t] - N
                
#             if flag == 1:
#                 if not isinf( states[t] ):
#                     st = <int>(states[t])
#                     helpMat = _logsumexp(log_forw[:,t])
#                     log_forw[:,t] = -INFINITY
#                     log_forw[st,t] = helpMat
                    
    return Ntsum



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.           
cpdef _log_backward2(dt[:,:]log_A, dt[:,:] log_p_states, dt[:,:] log_backw, 
                     dt[:] states, dt[:,:] log_lab,
                                         int T, int K):
    
    cdef int i
    cdef int t
    cdef int j, ii
    cdef dt[:] work_buffer = np.zeros(shape = [K])
    
    with nogil:
        for i in prange(K):
            log_backw[i,T-1] = 0
            
        
        for t in range(T-2, -1, -1):
            for i in range(K):
                if  isinf( states[t] ):
                    ii = 0
                else:
                    ii = <int>(states[t]) + 1
                   
                    
                for j in prange(K):
                    work_buffer[j] = log_A[i,j] + log_backw[j,t+1] + \
                                                        log_p_states[j,t+1] + log_lab[j, ii]
                
                log_backw[i,t] = _logsumexp(work_buffer)  
            


