#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:34:24 2020

@author: george
"""

import sys
sys.path.append('../')
sys.path.append('helper_functions/')
import  numpy as np
from _experiments import gauss_seq
from _misc import   dont_drop
from HMMs import MHMM
import matplotlib.pyplot as plt
from helper_functions.series_proc import relabel2 
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
import pandas as pd
from helper_functions.printer import print_summaries
from helper_functions.time_series_evaluations import time_series_multiple_th
from helper_functions.make_experiments import get_experiment
import time



np.random.seed( seed = 0)

#Initialize Transition Matrix
# a0 = [0.8, 0.2, 0]
# a1 = [0.2, 0.6, 0.2]
# a2 = [0, 0.2, 0.8]

a0 = [0.999, 0.001, 0]
a1 = [0.1, 0.6, 0.3]
a2 = [0, 0.4, 0.6]

#Initialize Means and Standard Deviations
mean = np.array([[-2,-2], [0,0], [2,2]])
std = np.array([[1,1], [1,1], [1,1]])
pi = [1,0,0]
A = np.array([a0, a1, a2])

#Initialize Time T and Number of Time Series N
T = 300
N = 1000

hmm_list  = []
mean_list = []
std_list  = []
m1 = [0,0]
m0 = [[x1, x1] for x1 in np.arange(-5, 0, 0.2)]
m2 = [[x1, x1] for x1 in np.arange(5, 0, -0.2 )]


for ii, (mm0, mm2) in enumerate(zip(m0,m2)):
    
    #GENERATE DATA
    mean = np.array([mm0,m1,mm2])
    mean_list.append(mean)
    std_list.append(std)
    data, states = gauss_seq(T = T, N = N, 
                          A = A, mean = mean, 
                          std = std, pi = pi)
    
    
   # data_ts, states_ts = gauss_seq(T = T, N = N, 
   #                                A = A, mean = mean,
   #                                std = std, pi = pi)    



    #train HMMM
    dates = None
    save_name = None
    states_off = 0
    n_HMMS = 1
    n_Comp = 1
    EM_iter = 500 #30
    tol = 10**(-10)
    n_states = 3

    A_mat = np.array([[0.6, 0.4, 0], [0.2, 0.7, 0.1],[ 0, 0.4, 0.6]])
    #A_mat = None

    pi = np.array([1, 0, 0])
    #pi = None

    #Preprocess States drop labels
    drop_perc = 0.7
    drop_list = [0.99, 0.99, 0.01]
    values = [0, 1, 2]
    states1 = dont_drop(values = values, states = states.copy(),
                    drop_perc = drop_perc, drop_list = drop_list)
    #states1 = states
    #states1 = None

    e =  10**(-7)
    label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,0], [0,e,e, 1-3*e]] )
    #label_mat = None

    inputs_class = {'n_HMMS':n_HMMS, 'n_states': n_states, 'n_Comp':n_Comp, 'EM_iter':EM_iter,
                'tol': tol}

    inputs_fit =  {'data': data, 'states': states1, 'dates' : dates, 'save_name': save_name, 
               'states_off': states_off, 'label_mat':label_mat, 'A': A_mat, 'pi': pi}

    start = time.time()

    mhmm = MHMM(**inputs_class)
    mhmm = mhmm.fit(**inputs_fit)

    print('Took:{}s'.format(time.time() - start))


    #get the hmm
    print('HMM {} trained'.format(ii))
    hmm = mhmm.HMMS[0]
    hmm_list.append(hmm)







                  

