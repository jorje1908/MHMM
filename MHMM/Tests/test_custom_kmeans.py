#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:36:17 2019

test custom kmeans

@author: george
"""

import sys
sys.path.append('../')
from _kmeans import custom_kmeans
from sklearn.cluster import KMeans



import  numpy as np
from _experiments import gauss_seq1d
from _misc import make_supervised, make_supervised2

np.random.seed( seed = 0)

#GENERATE DATA
a0 = [0.9, 0.1]
a1 = [0.6, 0.4]

m_0 = 0
m_1 = 3
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

#make supervised
#states1 = make_supervised(states.copy(), value = 0)

#make supervised 2
states1 = make_supervised2(states.copy())


states1d = states1.reshape(N*T)
data1d = data.reshape((N*T,-1))
K = 4
itera = 100
tol = 10**(-12)
m,indx_new, iner, m_st, m_o= custom_kmeans(data = data1d, labels = states1d, K = K, 
                      itera = itera, tol = tol)



kmeans = KMeans(n_clusters = K, random_state = 0)
kmeans = kmeans.fit(data1d)
m2 = kmeans.cluster_centers_
iner1 = kmeans.inertia_

print(iner, iner1)


