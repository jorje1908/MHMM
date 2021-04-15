#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:02:16 2020

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
from helper_functions.time_series_evaluations_new import  summary_evaluations_pd
import time


###############################################################################
#Part 1: 
# %%Data Generation
np.random.seed( seed = 5)

#Initialize Transition Matrix
# a0 = [0.999, 0.001, 0]
# a1 = [0.1, 0.6, 0.3]
# a2 = [0, 0.4, 0.6]

a0 = [0.999, 0.001, 0]
a1 = [0.05, 0.65, 0.3]
a2 = [0, 0.1, 0.9]

#Initialize Means and Standard Deviations
mean = np.array([[-2,-2], [0,0], [2,2]])
std = np.array([[1,1], [1,1], [1,1]])
pi = [1,0,0]
A = np.array([a0, a1, a2])

#Initialize Time T and Number of Time Series N
T = 300
N = 1000

#Generate 3d Data and 2d States Matrix
data, states = gauss_seq(T = T, N = N, A = A, mean = mean, std = std, pi = pi)
data_ts, states_ts = gauss_seq(T = T, N = N, A = A, mean = mean, std = std)                      


#dates = np.zeros( shape = [N, 2])
#dates[:,0] = np.random.choice( np.arange(8), size = N)
#dates[:,1] = np.random.choice( np.arange(8, 15), size = N)
###############################################################################
# %%Training
#Part 2:  Hmm Training
#TRAIN HMM
dates = None
save_name = None
states_off = 0
n_HMMS = 1
n_Comp = 1
EM_iter = 600 #30
tol = 10**(-10)
n_states = 3

A_mat = np.array([[0.6, 0.4, 0], [0.2, 0.7, 0.1],[ 0, 0.4, 0.6]])
#A_mat = None

pi = np.array([1, 0, 0])
#pi = None

#Preprocess States drop labels
drop_perc = 0.7
#drop_list = [0.99, 0.99, 0.01]
drop_list  = [1, 1, 0]
values = [0,1,2]
states1 = dont_drop(values = values, states = states.copy(),
                    drop_perc = drop_perc, drop_list = drop_list)
#states1 = states
#states1 = None

e =  10**(-7)*0
label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,0.0], [0,e,e, 1-3*e]] )
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
hmm = mhmm.HMMS[0]

#Get some Hmm Parameters
params = hmm.get_params()
cov = params['cov'].reshape(-1, mean.shape[1], mean.shape[1])

###############################################################################
# %%Relabelling
func = 'gmas'
data2d_pd   = relabel2(data, hmm,  labels_hmm = states1, 
                     labels = states, label_mat = label_mat, func = func)

data2d_pdts = relabel2(data_ts, hmm,  labels_hmm = None, 
                     labels = states_ts, label_mat = None, func = func)


sel       = -n_states-2
threshold = 0.95
xtrain    = data2d_pd.iloc[:,0:sel].values
ytrain_pd = data2d_pd.iloc[:, sel:]



xtest    = data2d_pdts.iloc[:,0: sel].values
ytest_pd =  data2d_pdts.iloc[:, sel:]


# %% Printing
#take combinations of the labels
if label_mat is not None:
    print("With labeling matrix")

print("For Training Data")
print_summaries(data = ytrain_pd, states = n_states, states2d = states)

print("\nFor Testing Data")

print_summaries(data = ytest_pd, states = n_states, states2d = states_ts)

# %% Supervised Training
C = 0.5

#Hmm relabeled multiclass classification
Xmd0 = xtrain.copy()
ymd0 = np.argmax(ytrain_pd[['state0', 'state1', 'state2']].values, axis = 1)
md0  = LogisticRegression(C = C)
md0  = md0.fit(Xmd0, ymd0)

#Hmm relabed  0s vs combinations of 1s and 2s 
thmd1      = [0.4, 0.4, 0.4]
Xmd1, ymd1 = get_experiment(X = xtrain, y = ytrain_pd, option = 1)
md1        = LogisticRegression(C = C)
md1        = md1.fit(Xmd1, ymd1.values)


#0s versus 2s 
thmd2      = [0.5, 0.5, 0.5]
Xmd2, ymd2 = get_experiment(X = xtrain, y = ytrain_pd, option = 2)
md2        = LogisticRegression(C = C)
md2        = md2.fit(Xmd2, ymd2.values)


# 0s versus  1s
thmd3      = [0.5, 0.5, 0.5]
Xmd3, ymd3 = get_experiment(X = xtrain, y = ytrain_pd, option = 3)
md3        = LogisticRegression(C = C)
md3        = md3.fit(Xmd3, ymd3.values)


# 0s versus  1s, drop 2s bigger than threshold
thmd4      = [0.2, 0.2, 0.6]
Xmd4, ymd4 = get_experiment(X = xtrain, y = ytrain_pd, option = 4)
md4        = LogisticRegression(C = C)
md4        = md4.fit(Xmd4, ymd4.values)


# 0s +1s versus 1s+2s, 
thmd5      = [0.2, 0.2, 0.2]
Xmd5, ymd5 = get_experiment(X = xtrain, y = ytrain_pd, option = 5)
md5        = LogisticRegression(C = C)
md5        = md5.fit(Xmd5, ymd5.values)







#Baseline 0s versus 2s unrealizable
Xb1 = xtrain.copy()
yb1 = (ytrain_pd.labels >= 2).astype(int)
md6 = LogisticRegression(C = C)
md6 = md6.fit(Xb1, yb1.values)

#Baseline 0s versus 2s where we have missing data
Xb2 = xtrain.copy()
yb2 = (ytrain_pd.labels_hmm >= 2).astype(int)
md7 = LogisticRegression(C = C)
md7 = md7.fit(Xb2, yb2)







# %%get test probabilities
Nts = len(data_ts)

#multi class classification
prob_test0 =  np.sum(md1.predict_proba(xtest)[:,1:], 
                     axis = 1).reshape([Nts,T])

#Hmm relabed  0s vs combinations of 1s and 2s 
prob_test1 = md1.predict_proba(xtest)[:,1].reshape([Nts,T])

#0s versus 2s 
prob_test2 = md2.predict_proba(xtest)[:,1].reshape([Nts,T])

#0s versus 1s
prob_test3 = md3.predict_proba(xtest)[:,1].reshape([Nts,T])

# 0s +1s versus  2s, 
prob_test4 = md4.predict_proba(xtest)[:,1].reshape([Nts,T])

# 0s + 1s versus  1s + 2s, 
prob_test5 = md5.predict_proba(xtest)[:,1].reshape([Nts,T])

#Baseline 0s versus 2s unrealizable
prob_test6 = md6.predict_proba(xtest)[:,1].reshape([Nts,T])

#Baseline 0s versus 2s where we have missing data
prob_test7 = md7.predict_proba(xtest)[:,1].reshape([Nts,T])



#get optimal predictor (that knows the states)
prob_optimal = states_ts.copy()
np.place(prob_optimal, prob_optimal == 2, 1)


# %%evaluation
#get test data 0-1s

forward_horizon = 50
backward_horizon = 0
lookback = 2
taus = 10
tau_list = [x for x in np.arange(0, 1, 0.05)]

kwargs = {'forward_horizon': forward_horizon, 
          'backward_horizon': backward_horizon, 
          'tau_list': tau_list
          }
ytest = (ytest_pd.labels.values>=2).astype(int).reshape([Nts, T])


ev0  =   summary_evaluations_pd(probabilities = prob_test0, labels =  ytest, 
                              **kwargs)
#evaluation of hmm relabeled
ev1, mets_pd  = summary_evaluations_pd(probabilities = prob_test1, labels =  ytest, 
                              **kwargs, ts_metrics = True)

#evaluation of second hmm relabeled model optimal
ev2  =  summary_evaluations_pd(probabilities = prob_test2, labels =  ytest,
                               **kwargs)
   
#evaluation of baseline                              
ev3  = summary_evaluations_pd(probabilities = prob_test3, labels =  ytest, 
                              **kwargs)

#evaluation of 2nd bseline
ev4  = summary_evaluations_pd(probabilities = prob_test4, labels =  ytest,
                             **kwargs)

#evaluation of 3rd relabeled
ev5  = summary_evaluations_pd(probabilities = prob_test5, labels =  ytest,
                              **kwargs)

#
ev6  = summary_evaluations_pd(probabilities = prob_test6, labels =  ytest, 
                              **kwargs)

#
ev7  = summary_evaluations_pd(probabilities = prob_test7, labels =  ytest,
                             **kwargs)


ev_opt,_ =  time_series_multiple_th(prob_optimal, ytest,
                                    taus = taus, lookback = lookback)


# %% display evaluations
#evaluation of optimal




pd.set_option('display.max_columns', 40)

print("\nHMM 0s versus 1s plus 2s all")
display(ev0)

print("\nHMM Relabeled 0s versus combination of 1s and 2s")
display(ev1)

print('\nHmm relabeled 0s versus 2s')
display(ev2)

print("\nrelabeld 0s versus 1s")
display(ev3)

print('\n0s +  1s, versus 2s')
display(ev4)

print("\n0s +  1s, versus 1s+2s")
display(ev5)

print("\nHMM Baseline 0")
display(ev6)

print("\nHMM Baseline 1")
display(ev7)

print('\nS tate Predictor Classifier')
display(ev_opt)





# %%Plotting
ytrain_pd['labels2'] = (ytrain_pd.labels >=1).astype(int)
mask = ytrain_pd.labels2 == 1
x = np.random.uniform(low = -15, high = 15, size = 100000)

coef1 = md1.coef_[0]
int1 = md1.intercept_[0]
x21 = -x*coef1[0]/coef1[1] +(- int1  )/coef1[1]

coef2 = md2.coef_[0]
int2 = md2.intercept_[0]
x22 = -x*coef2[0]/coef2[1] + (-int2 )/coef2[1]

coef3 = md3.coef_[0]
int3 = md3.intercept_[0]
x33 = -x*coef3[0]/coef3[1] + (-int3 )/coef3[1]

coef4 = md4.coef_[0]
int4 = md4.intercept_[0]
x44 = -x*coef4[0]/coef4[1] + (-int4 )/coef4[1]

coef5 = md5.coef_[0]
int5 = md5.intercept_[0]
x55 = -x*coef5[0]/coef5[1] + (-int5 )/coef5[1]


data2d = data.reshape([N*T, -1])
fig, ax = plt.subplots(1,1, figsize = (12,12))
ax.scatter(data2d[mask][:,0], data2d[mask][:,1], s = 0.1)
ax.scatter(data2d[~mask][:,0], data2d[~mask][:,1], s = 0.1)

ax.plot(x, x21)
ax.plot(x, x22)
ax.plot(x, x33)
ax.plot(x, x44)
ax.plot(x, x55)


ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(['Hmm 0 vs 1,2', 'Hmm 0 vs 1', 'Baseline  0 vs 2', 
           'Baseline 0 vs 2 Missing', 'Hmm 0 vs 1,2 missing'])
ax.set_title('Lines')
#ax.set_xlim(xmin = -10,xmax = 10)



# %% Multilabel Classification train
y_mult = np.argmax(ytrain_pd[['state0', 'state1', 'state2']].values, axis = 1)
X_mult = xtrain

lg_mult = LogisticRegression(tol = 10**(-8), penalty = 'l2')
lg_mult = lg_mult.fit(X_mult, y_mult)

# %% Multilabel Classification evaluate
prob_mult = np.sum(lg_mult.predict_proba(X_mult)[:,1:], axis = 1)
prob_mult2d = prob_mult.reshape([N,T])

mask_mul = (states >= 2).any(axis = 1)
states_pos = states[mask_mul]
states_pos2 = np.where(states_pos == 2, 2, 0)
prob_mul2dpos = prob_mult2d[mask_mul]


k = 98
fig, ax = plt.subplots(1,1)
ax.plot(states_pos2[k])
ax.plot(prob_mul2dpos[k])


# %% evaluations
import pickle
local_vars = dict(locals())


evaluations_ts = [local_vars['ev'+str(i)] for i in range(8)]

pickle.dump(evaluations_ts, open('early_prediction_results/evals.p', 'wb'))

