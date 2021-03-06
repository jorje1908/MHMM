#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:53:57 2019

@author: george

gaussian experiment for HMMS

"""

import sys
sys.path.append('../')
import  numpy as np
from _experiments import gauss_seq1d
from _misc import make_supervised, compute_forw, make_supervised2
from HMMs import MHMM
import matplotlib.pyplot as plt

np.random.seed( seed = 0)

#GENERATE DATA
a0 = [0.9, 0.1]
a1 = [0.4, 0.6]

m_0 = 0
m_1 = 4
std_0 = 1
std_1 = 1

A = np.array([a0, a1])
T = 15
N = 1000

data, states = gauss_seq1d(T = T, N = N, A = A, m_0 = m_0, m_1 = m_1,
                           std_0 = std_0, std_1 = std_1)


dates =np.zeros( shape = [N, 2])
dates[:,0] = np.random.choice( np.arange(8), size = N)
dates[:,1] = np.random.choice( np.arange(8, 15), size = N)

#TRAIN HMM
n_HMMS = 1
n_Comp = 1
EM_iter = 2

#states1 = make_supervised(states.copy(), value = 0)
states1 = make_supervised2(states.copy(), drop = 0)
#statesinf = np.full( shape = [states1.shape[0], states1.shape[1]], fill_value = -np.inf )
#statesinf[0, 10] = 1

mhmm = MHMM(n_HMMS = n_HMMS, n_states = 2, n_Comp = n_Comp, EM_iter = EM_iter, tol = 10**(-5))
mhmm = mhmm.fit( data = data, states = states1, dates = None, save_name = 'mymhmm.npy')

#get the hmm
hmm = mhmm.HMMS[0]
params = hmm.get_params()


zers, ones = compute_forw(hmm, data)

fig, ax = plt.subplots(1,3, figsize = (10,4))
ax[0].hist(zers, bins = 30)
ax[1].hist(ones, bins = 30)
ax[2].hist(np.concatenate((ones, zers), axis = 0), bins = 30)
ax[2].set_title('All states1')
