#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 02:09:10 2020

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

def hmm_cases():
        a0 = [0.8, 0.1, 0.1]
        a1 = [0.2, 0.7, 0.1]
        a2 = [0.3, 0.2, 0.5]
        
        #Initialize Means and Standard Deviations
        mean = np.array([[-0.5, -0.5], [0, 0], [0.5, 0.5]])
        std = np.array([[1,1], [1,1], [1,1]])
        pi2 = [0.6,0.2,0.2]
        A = np.array([a0, a1, a2])
        
        #Initialize Time T and Number of Time Series N
        T = 300
        N = 1000
        
        #Generate 3d Data and 2d States Matrix
        data, states = gauss_seq(T = T, N = N, A = A, mean = mean, std = std, pi = pi2)
        states_fl    = states.flatten()
        zeros        = np.sum(states_fl == 0)/(N*T)
        ones         = np.sum(states_fl == 1)/(N*T)
        twos         = np.sum(states_fl == 2)/(N*T)
        print("Zeros:{}, Ones:{}, Twos:{}".format(zeros, ones, twos))
        
        
        #general HMM parametres
        dates = None
        save_name = None
        states_off = 0
        n_HMMS = 1
        n_Comp = 1
        EM_iter = 600#30
        tol = 10**(-10)
        n_states = 3
        
        A_mat = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1],[ 0.2, 0.4, 0.2]])
        pi = np.array([0.4, 0.4, 0.2])
        #A_mat = None
        
        A_fl    = A.flatten()
        pi_fl   = np.array(pi2).flatten()
        mean_fl = mean.flatten()
        ss      = np.array([[1,0], [0,1]])
        std_fl  = list(ss.flatten())
        std_fl  = np.array(std_fl*3)
        #pi = None
        
        #Preprocess States drop labels
        # drop_perc = 0.7
        # drop_list = [0.99, 0.99, 0.01]
        # values = [0,1,2]
        # states1 = dont_drop(values = values, states = states.copy(),
        #                     drop_perc = drop_perc, drop_list = drop_list)
        #states1 = states
        
        errors = []
        # %% Case 1 Unsupervised Training
        states1 = None
        
        e =  10**(-7)
        #label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,0.0], [0,e,e, 1-3*e]] )
        label_mat = None
        
        inputs_class = {'n_HMMS':n_HMMS, 'n_states': n_states, 'n_Comp':n_Comp, 'EM_iter':EM_iter,
                        'tol': tol}
        
        inputs_fit =  {'data': data, 'states': states1, 'dates' : dates, 'save_name': save_name, 
                       'states_off': states_off, 'label_mat':label_mat, 'A': A_mat, 'pi': pi}
        
        start = time.time()
        
        mhmm1 = MHMM(**inputs_class)
        mhmm1 = mhmm1.fit(**inputs_fit)
        
        print('Took:{}s'.format(time.time() - start))
        
        # %% Case 1  Errors
        #get the hmm
        indx = [0,2,1]
        hmm_1 = mhmm1.HMMS[0]
        hmm_par1 = hmm_1.get_params()
        A_1 = hmm_1.A[indx,:][:,indx].flatten()
        means_1 = hmm_1.means[indx].flatten()
        cov_1 = hmm_1.cov[indx].flatten()
        pi_1 = hmm_1.pi[indx].flatten()
        
        
        A_err1 = np.sum(np.abs(A_1-A_fl))
        means_err1 = np.sum(np.abs(means_1-mean_fl))
        cov_err1 = np.sum(np.abs(cov_1-std_fl))
        pi_err1 = np.sum(np.abs(pi_1-pi_fl))
        
        
        errors.append([A_err1, means_err1, cov_err1, pi_err1])
        
        
        # %%Case 2 fully Supervised
        
        states2 = states.copy()
        
        e =  10**(-7)
        #label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,0.0], [0,e,e, 1-3*e]] )
        label_mat = np.log( [[0,1,0,0], [0,0,1,0], [0,0,0,1]] )
        
        inputs_class = {'n_HMMS':n_HMMS, 'n_states': n_states, 'n_Comp':n_Comp, 'EM_iter':EM_iter,
                        'tol': tol}
        
        inputs_fit =  {'data': data, 'states': states2, 'dates' : dates, 'save_name': save_name, 
                       'states_off': states_off, 'label_mat':label_mat, 'A': A_mat, 'pi': pi}
        
        start = time.time()
        
        mhmm2 = MHMM(**inputs_class)
        mhmm2 = mhmm2.fit(**inputs_fit)
        
        print('Took:{}s'.format(time.time() - start))
        
        # %% Case 2  Errors
        #get the hmm
        hmm_2 = mhmm2.HMMS[0]
        hmm_p2 = hmm_2.get_params()
        A_2 = hmm_2.A.flatten()
        means_2 = hmm_2.means.flatten()
        cov_2 = hmm_2.cov.flatten()
        pi_2 = hmm_2.pi.flatten()
        
        
        A_err2 = np.sum(np.abs(A_2-A_fl))
        means_err2 = np.sum(np.abs(means_2-mean_fl))
        cov_err2 = np.sum(np.abs(cov_2-std_fl))
        pi_err2 = np.sum(np.abs(pi_2-pi_fl))
        
        
        errors.append([A_err2, means_err2, cov_err2, pi_err2])
        
        
        # %%Case 3 Drop Randomly
        #drop_list = [0.99, 0.99, 0.01]
        #values = [0,1,2]
        dp = zeros*0.99 + ones*0.99 + twos*0.8
        states3 = dont_drop(values = None, states = states.copy(),
                            drop_perc = dp, drop_list = None)
        #states1 = states
        
        e =  10**(-7)
        #label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,0.0], [0,e,e, 1-3*e]] )
        label_mat = np.log( [[dp,1-dp,0,0], [dp,0,1-dp,0], [dp,0,0,1-dp]] )
        
        inputs_class = {'n_HMMS':n_HMMS, 'n_states': n_states, 'n_Comp':n_Comp, 'EM_iter':EM_iter,
                        'tol': tol}
        
        inputs_fit =  {'data': data, 'states': states3, 'dates' : dates, 'save_name': save_name, 
                       'states_off': states_off, 'label_mat':label_mat, 'A': A_mat, 'pi': pi}
        
        start = time.time()
        
        mhmm3 = MHMM(**inputs_class)
        mhmm3 = mhmm3.fit(**inputs_fit)
        
        print('Took:{}s'.format(time.time() - start))
        
        # %% Case 3 Errors
        #get the hmm
        hmm_3 = mhmm3.HMMS[0]
        hmm_p3 = hmm_3.get_params()
        A_3 = hmm_3.A.flatten()
        means_3 = hmm_3.means.flatten()
        cov_3 = hmm_3.cov.flatten()
        pi_3 = hmm_3.pi.flatten()
        
        
        A_err3 = np.sum(np.abs(A_3-A_fl))
        means_err3 = np.sum(np.abs(means_3-mean_fl))
        cov_err3 = np.sum(np.abs(cov_3-std_fl))
        pi_err3 = np.sum(np.abs(pi_3-pi_fl))
        
        
        errors.append([A_err3, means_err3, cov_err3, pi_err3])
        
        
        # %%Case 4 Drop biased correct labeling matrix
        drop_list = [0.99, 0.99, 0.8]
        values = [0,1,2]
        states4 = dont_drop(values = values, states = states.copy(),
                            drop_perc = 0, drop_list = drop_list)
        #states1 = states
        
        e =  10**(-7)
        #label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,0.0], [0,e,e, 1-3*e]] )
        label_mat = np.log( [[0.99,0.01,0,0], [0.99,0,0.01,0], [0.2,0,0,0.8]] )
        
        inputs_class = {'n_HMMS':n_HMMS, 'n_states': n_states, 'n_Comp':n_Comp, 'EM_iter':EM_iter,
                        'tol': tol}
        
        inputs_fit =  {'data': data, 'states': states4, 'dates' : dates, 'save_name': save_name, 
                       'states_off': states_off, 'label_mat':label_mat, 'A': A_mat, 'pi': pi}
        
        start = time.time()
        
        mhmm4 = MHMM(**inputs_class)
        mhmm4 = mhmm4.fit(**inputs_fit)
        
        print('Took:{}s'.format(time.time() - start))
        
        # %% Case 4  Errors
        #get the hmm
        hmm_4 = mhmm4.HMMS[0]
        hmm_p4 = hmm_4.get_params()
        A_4 = hmm_4.A.flatten()
        means_4 = hmm_4.means.flatten()
        cov_4 = hmm_4.cov.flatten()
        pi_4 = hmm_4.pi.flatten()
        
        
        A_err4 = np.sum(np.abs(A_4-A_fl))
        means_err4 = np.sum(np.abs(means_4-mean_fl))
        cov_err4 = np.sum(np.abs(cov_4-std_fl))
        pi_err4 = np.sum(np.abs(pi_4-pi_fl))
        
        
        errors.append([A_err4, means_err4, cov_err4, pi_err4])
        
        # %%Case 5 Drop biased unif relab
        drop_list = [0.99, 0.99, 0.8]
        values = [0,1,2]
        states5 = dont_drop(values = values, states = states.copy(),
                            drop_perc = 0, drop_list = drop_list)
        
        dp = zeros*0.99 + ones*0.99 + twos*0.8
        #states1 = states
        
        e =  10**(-7)
        #label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,0.0], [0,e,e, 1-3*e]] )
        label_mat = np.log( [[dp,1-dp,0,0], [dp,0,1-dp,0], [dp,0,0,1-dp]] )
        
        inputs_class = {'n_HMMS':n_HMMS, 'n_states': n_states, 'n_Comp':n_Comp, 'EM_iter':EM_iter,
                        'tol': tol}
        
        inputs_fit =  {'data': data, 'states': states5, 'dates' : dates, 'save_name': save_name, 
                       'states_off': states_off, 'label_mat':label_mat, 'A': A_mat, 'pi': pi}
        
        start = time.time()
        
        mhmm5 = MHMM(**inputs_class)
        mhmm5 = mhmm5.fit(**inputs_fit)
        
        print('Took:{}s'.format(time.time() - start))
        
        # %% Case 5 Errors
        #get the hmm
        hmm_5 = mhmm5.HMMS[0]
        hmm_p5 = hmm_5.get_params()
        A_5 = hmm_5.A.flatten()
        means_5 = hmm_5.means.flatten()
        cov_5 = hmm_5.cov.flatten()
        pi_5 = hmm_5.pi.flatten()
        
        
        A_err5 = np.sum(np.abs(A_5-A_fl))
        means_err5 = np.sum(np.abs(means_5-mean_fl))
        cov_err5 = np.sum(np.abs(cov_5-std_fl))
        pi_err5 = np.sum(np.abs(pi_5-pi_fl))
        
        
        errors.append([A_err5, means_err5, cov_err5, pi_err5])
        
        
        return np.array(errors)
    
    
np.random.seed(seed = 5) 
for i in range(5):
    print(i)

    if i == 0:
        er = hmm_cases()
    else:
        er1 = hmm_cases()
        er += er1
        
er = er/5
            