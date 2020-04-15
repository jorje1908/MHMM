#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:58:36 2020

@author: george
"""
# %%imports
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
###############################################################################
#Part 1: 
# %%Data Generation
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

#Generate Data and States
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
EM_iter = 120 #30
tol = 10**(-9)
n_states = 3

A_mat = np.array([[0.6, 0.4, 0], [0.2, 0.7, 0.1],[ 0, 0.4, 0.6]])
#A_mat = None

pi = np.array([1, 0, 0])
#pi = None

#Preprocess States drop labels
drop_perc = 0.7
states1 = dont_drop(values = 2, states = states.copy(), drop_perc = drop_perc)
#states1 = states
#states1 = None

e =  10**(-7)
label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,e], [e,e,e, 1-3*e]] )
#label_mat = None

inputs_class = {'n_HMMS':n_HMMS, 'n_states': n_states, 'n_Comp':n_Comp, 'EM_iter':EM_iter,
                'tol': tol}

inputs_fit =  {'data': data, 'states': states1, 'dates' : dates, 'save_name': save_name, 
               'states_off': states_off, 'label_mat':label_mat, 'A': A_mat, 'pi': pi}

mhmm = MHMM(**inputs_class)
mhmm = mhmm.fit(**inputs_fit)

#get the hmm
hmm = mhmm.HMMS[0]

#Get some Hmm Parameters
params = hmm.get_params()
cov = params['cov'].reshape(-1, mean.shape[1], mean.shape[1])

###############################################################################
# %%Relabelling
func = 'gmas'
data2d_pd = relabel2(data, hmm,  labels_hmm = states1, 
                     labels = states, label_mat = label_mat, func = func)

data2d_pdts = relabel2(data_ts, hmm,  labels_hmm = None, 
                     labels = states_ts, label_mat = None, func = func)

sel = -n_states-2
threshold = 0.95
trainf = data2d_pd.iloc[:,0:sel]
pl = data2d_pd.iloc[:, sel:]

#choose the labels for training
pl['labels2'] = (pl.labels >= 2).astype(int)
pl['sum12'] = ((pl.state1 + pl.state2) > threshold).astype(int)
pl['hmm'] = (pl.labels_hmm == 2).astype(int)

testf = data2d_pdts.iloc[:,0: sel]
pl_ts =  data2d_pdts.iloc[:, sel:]
pl_ts['labels2'] = (pl_ts.labels >= 2).astype(int)
pl_ts['sum12'] = ((pl_ts.state1 + pl.state2) > threshold).astype(int)


# %% Printing
#take combinations of the labels
if label_mat is not None:
    print("With labeling matrix")

print("For Training Data")
print_summaries(data = pl, states = n_states, states2d = states)

print("\nFor Testing Data")

print_summaries(data = pl_ts, states = n_states, states2d = states_ts)

# %% Supervised Training
C = 0.5

#Hmm relabed  0s vs combinations of 1s and 2s 
md1 = LogisticRegression(C = C)
md1 = md1.fit(trainf.values, pl.sum12.values)


#0s versus 1s
pl['new'] = 0
pl['new'][pl.state1 >= 0.5] = 1

th = [0.5, 0.5]
ylabels_0_vs_1 = pl[(pl.state0 >= th[0]) | (pl.state1 >= th[1])].new

train_0_vs_1 = trainf[(pl.state0 >= th[0]) | (pl.state1 >= th[1])].values

#Hmm relabeld 0 vesrus 1
md2 = LogisticRegression()
md2 = md2.fit(train_0_vs_1, ylabels_0_vs_1.values)


#Baseline 0s versus 2s
md3 = LogisticRegression(C = C)
md3 = md3.fit(trainf.values, pl.labels2.values)

#Baseline 0s versus 2s where we have missing data
md4 = LogisticRegression(C = C)
md4 = md4.fit(trainf.values, pl.hmm.values)


#hmm relabeled 0s vs combinations of 1s and 2s missing
threshold_comb = 0.95
pl['new2'] = 0
pl['new2'] = ((pl.state1 + pl.state2) > threshold_comb).astype(int)
th2 = [0.7, 0.95]
mask = (pl.state0 >= th2[0]) | ((pl.state1 + pl.state2) >= th2[1])
ylabels_0_vs_12_comb = pl[mask].new2
train_0_vs_12 = trainf[mask].values

md5 = LogisticRegression(C = C)
md5 = md5.fit(train_0_vs_12, ylabels_0_vs_12_comb)




# %%evaluation
Nts = len(data_ts)

#get probabilities for hmm relabeled model 
prob_test = md1.predict_proba(testf.values)[:,1].reshape([Nts,T])

#get probabilities for the baseline 0 versus 1 hmm relabeled
prob_test2 = md2.predict_proba(testf.values)[:,1].reshape([Nts,T])

#get probabilities for baseline model 0s versus 2s
prob_test3 = md3.predict_proba(testf.values)[:,1].reshape([Nts,T])

#get probabilities for baseline model 0s versus 2s
prob_test4 = md4.predict_proba(testf.values)[:,1].reshape([Nts,T])

#get probabilities for hmm relabeled model 0s versus combination 1s and 2s missing
prob_test5 = md5.predict_proba(testf.values)[:,1].reshape([Nts,T])

#get optimal predictor (that knows the states)
prob_optimal = states_ts.copy()
np.place(prob_optimal, prob_optimal == 2, 1)

#get test data 0-1s
ytest = pl_ts.labels2.values.reshape([Nts, T])

lookback = 0
taus = 26
#evaluation of hmm relabeled
ev1, _ = time_series_multiple_th(prob_test, ytest, taus = taus, lookback = lookback)

#evaluation of second hmm relabeled model optimal
ev2,_ =  time_series_multiple_th(prob_test2, ytest, taus = taus, lookback = lookback)
  
#evaluation of baseline                              
ev3, _ = time_series_multiple_th(prob_test3, ytest, taus = taus, lookback = lookback)

#evaluation of 2nd bseline
ev4, _ = time_series_multiple_th(prob_test4, ytest, taus = taus, lookback = lookback)

#evaluation of 3rd relabeled
ev5, _ = time_series_multiple_th(prob_test5, ytest, taus = taus, lookback = lookback)


#evaluation of optimal
ev_opt,_ =  time_series_multiple_th(prob_optimal, ytest, taus = taus, lookback = lookback)




pd.set_option('display.max_columns', 20)
print("\nHMM Relabeled 0s versus combination of 1s and 2s")
display(ev1)

print('\nHmm relabeled 0s versus 1s')
display(ev2)

print("\nBaseline Classifier 0s and 2s")
display(ev3)

print('\nSecond Baseline Missing Data')
display(ev4)

print("\nHMM Relabeled 0s versus combination of 1s and 2s, missing")
display(ev5)

print('\nState Predictor Classifier')
display(ev_opt)





# %%Plotting
mask = pl.labels2 == 1
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
ax.set_title('gs_exp3_2')
#ax.set_xlim(xmin = -10,xmax = 10)



# %% CLassification report
ypred1 = (prob_test >= 0.5).reshape(Nts*T)
ypred2 = (prob_test2 >= 0.5).reshape(Nts*T)

from sklearn.metrics import classification_report
print('\n', classification_report(pl_ts.labels2.values, ypred1))
print('\n', classification_report(pl_ts.labels2.values, ypred2))






