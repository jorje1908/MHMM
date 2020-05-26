#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:16:14 2020

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
import pickle

import matplotlib.pyplot as plt


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


m1 = [0,0]
m0 = [[x1, x1] for x1 in np.arange(-5, 0, 0.2)]
m2 = [[x1, x1] for x1 in np.arange(5, 0, -0.2 )]



hmmr_list  = pickle.load(open('trained_hmms/hmm_r.p','rb'))
hmm_list  = pickle.load(open('trained_hmms/hmm.p','rb'))

error_A = []
error_pi = []
error_m = []
error_cov = []

for hmms, hmmu, mm0,mm2 in zip(hmmr_list, hmm_list, m0, m2):
    
    
    mean = np.array([mm0,m1,mm2])
    means_fl  = mean.flatten()
    cov_fl   =  list(np.array([[1,0],[0,1]]).flatten())
    cov_fl   = np.array(cov_fl*3)
    A_fl     = A.flatten()
    pi_fl    = np.array([1, 0, 0])
    
    Asup, Auns  = np.abs(hmms.A.flatten()-A_fl), np.abs(hmmu.A.flatten()-A_fl)
    pisup,piuns = np.abs(hmms.pi.flatten()-pi_fl), np.abs(hmmu.pi.flatten()-pi_fl)
    msup,muns = np.abs(hmms.means.flatten()-means_fl), np.abs(hmmu.means.flatten()-means_fl)
    csup,cuns = np.abs(hmms.cov.flatten()-cov_fl), np.abs(hmmu.cov.flatten()-cov_fl)
    
    error_A.append([Asup.sum(), Auns.sum()])
    error_pi.append([pisup.sum(), piuns.sum()])
    error_m.append([msup.sum(), muns.sum()])
    error_cov.append([csup.sum(), cuns.sum()])
    
    
    
Aerr  = np.array(error_A).mean(axis = 0)
pierr = np.array(error_pi).mean(axis = 0)  
merr  = np.array(error_m).mean(axis = 0)
coverr = np.array(error_cov).mean(axis = 0)  






