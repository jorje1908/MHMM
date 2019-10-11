#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:25:26 2019

@author: george
"""

import numpy as np
import pandas as pd
import time

from _experiments import generate_Coin
from HMMs import MHMM
from _visu import plot_prob_state
np.random.seed( seed = 2 )


#generate coins
N = 1000
data, states, coins = generate_Coin(N = N)

#get partial states
st_part = np.full( shape = [1, N], fill_value = -np.inf)
st_part[0, 0] = states[0]

#initialize MHMM
model = MHMM(n_HMMS = 1, n_states = 2, EM_iter = 400, tol = 10**(-12), n_Comp = 1)

#fit HMM
start = time.time()
model = model.fit( data = data[1:] , states = states.T)
end = time.time() - start
print("time elapsed: {:.4}".format(end))

#get logliklihood
logLi = model.logLikehood

#get model's parameters
params = model.get_params()['params']

#take the fitted HMM
hmmM = model.HMMS[0]

#get gamas and forward and backward and posterior
log_gamas = hmmM.log_gamas( data[1] )
gamas = np.exp(log_gamas)

log_forw = hmmM.log_forward( data[1])
forw = np.exp(log_forw)

log_back = hmmM.log_backward( data[1])
back = np.exp( log_back )

posterior_x = hmmM.log_predict_x( data[1] )
#get max_states
gamma_states = np.argmax(log_gamas, axis = 0)

#get gamma 1 and gamma 0
log_gamma1 = log_gamas[1,:]
log_gamma0 = log_gamas[0,:]
gamma1 = np.exp(log_gamas[1,:])
gamma0 = np.exp(log_gamas[0,:])




#take the viterbi states
vit_log, vit_states, vit_seq, p_states = hmmM.log_viterbi( data[1])
vit = np.exp(vit_log)
#get viterbi 1 and 0
log_vit1 = vit_log[1,:]
log_vit0 = vit_log[0,:]
vit1 = np.exp(log_vit1)
vit0 = np.exp(log_vit0)

#get the accuracies 
accuracy_viterbi =  np.sum( vit_seq == states[:,0])/N
accuracy_gamma = np.sum( gamma_states == states[:,0])/N


#plotting
plot_prob_state(gamma0, 1,  np.arange(N), states, name = "smoothed")
plot_prob_state(vit0, 1,  np.arange(N), states, name = "viterbi")
plot_prob_state(forw[0,:], 1,  np.arange(N), states, name = "forward")





