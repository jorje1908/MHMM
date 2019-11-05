#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:53:57 2019

@author: george

gaussian experiment for HMMS

"""


import  numpy as np
from _experiments import gauss_seq1d
from _misc import make_supervised, compute_forw
from HMMs import MHMM
import matplotlib.pyplot as plt

np.random.seed( seed = 0)

#GENERATE DATA
a0 = [0.9, 0.1]
a1 = [0.4, 0.6]

m_0 = 0
m_1 = 3
std_0 = 1
std_1 = 1

A = np.array([a0, a1])
T = 15
N = 200

data, states = gauss_seq1d(T = T, N = N, A = A, m_0 = m_0, m_1 = m_1,
                           std_0 = std_0, std_1 = std_1)


dates =np.zeros( shape = [N, 2])
dates[:,0] = np.random.choice( np.arange(8), size = 200)
dates[:,1] = np.random.choice( np.arange(8, 15), size = 200)

#TRAIN HMM
n_HMMS = 1
n_Comp = 1
EM_iter = 20
states1 = make_supervised(states.copy(), value = 0)
#statesinf = np.full( shape = [states1.shape[0], states1.shape[1]], fill_value = -np.inf )
#statesinf[0, 10] = 1

mhmm = MHMM(n_HMMS = n_HMMS, n_Comp = n_Comp, EM_iter = EM_iter, tol = 10**(-3))
mhmm = mhmm.fit( data = data, states = None, dates = None)

#get the hmm
hmm = mhmm.HMMS[0]
params = hmm.get_params()


zers, ones = compute_forw(hmm, data)

fig, ax = plt.subplots(1,3, figsize = (10,4))
ax[0].hist(zers, bins = 30)
ax[1].hist(ones, bins = 30)
ax[2].hist(np.concatenate((ones, zers), axis = 0), bins = 30)
