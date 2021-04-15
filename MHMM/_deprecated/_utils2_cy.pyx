#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 20:50:11 2019

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
    
    
    
    
    
    
cpdef _log_predict_states_All( X ):
        """
        OBSERVATION MODEL  FOR ALL STATES
        
        computes the probability to observe a x_t for all t = 1...T
        for all states K = 1...K
        for any observation in time it calculates the probability 
        using the method log_predict_states
        
        X  = (T,d) is a sequence of observations x1, ... xT
        numpy array Txd
        
        return the matrix P[k,t] probability to observe x_t given state k
        
        p(x_t | z_t = k, H = m) for all t for all k
        
        P = [K,T] matrix
        
        """
        
        #get the number of sequneces in time
        T = X.shape[0]
        #get number of states
        K = self.states_
        #initialize the array
        log_P_all = np.zeros( shape = [K, T])
        
        for t in range( T ): #for every time step
           log_P_all[:, t] = self.log_predict_states( X[t] )
        
        return log_P_all
    
cpdef _log_predict_states( x):
    """ 
     calculates the probability of x to be observed given a state,
     for all states and returns a matrix 
     P(x) = [p(x_t|z_t = 1)... p(x_t|z_t = K)]
     
     x = (d, )
     
     retuns P(x) = (K,)
    
    """
    
    #take the number of states
    K = self.states_
    
    #initialize P
    log_P = np.zeros( shape = [K] )
    
    for k in np.arange( K ): #for every state 
        log_P[k] = self.log_predict_state(x, st = k)
     
   
    return log_P
    
    
cpdef _log_predict_state( x, st ):
     """
     calculates the probabiity of an observation to be generated 
     by the state "state" --> p(x_t | z_t = k)
     
     returns a matrix kx1
     x = (d,)
     
     """
     
     #get number of Gaussian Components
     components = self.g_components_
     
     #initialize probability to 0
     log_pk = -np.math.inf #was 0
     
     for component in np.arange( components ): #for every G compo
         log_pk = logaddexp(log_pk,
                _log_predict_state_comp( x, st , cmp ))
      
    
     return log_pk
        
     
     
cpdef log_predict_state_comp(self, x, st , cmp ):
     
     """ 
     predicts the probability of an observation to be generated
     by the lth component gaussian of the kth state
     
     st: HMM  state {1...K}
     cmp: Kth state lth Gaussian Component l = {1...L}
     
     p( G_t = l, x_t |  z_t = k) = 
     p(x_t | G_t = l, z_t = k)p(G_t = l|z_t = k)
     
     x = (d,)
     
     """
     
     #get the multivariate_gaussian of state = "state"
     #component = "comp"
     gaussian_kl = self.gauss[st][ cmp ]
     
     #get the probability of this component to be chosen
     log_alpha_kl = np.log(self.alpha[st, cmp])
     
     #predict the probability of x to be generated by the lth component
     pkl = gaussian_kl.logpdf( x ) + log_alpha_kl
     
     return pkl