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
from series_proc import relabel2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
from printer import print_summaries
from time_series_evaluations import time_series_multiple_th
###############################################################################
#Part 1: 
# %%Data Generation
np.random.seed( seed = 0)

#Initialize Transition Matrix
# a0 = [0.8, 0.2, 0]
# a1 = [0.2, 0.6, 0.2]
# a2 = [0, 0.2, 0.8]

a0 = [0.995, 0.005, 0]
a1 = [0.1, 0.4, 0.5]
a2 = [0, 0.3, 0.7]

#Initialize Means and Standard Deviations
mean = np.array([[-1,-1],[3,3], [4,4]])
std = np.array([[1,1],[1,1], [1,1]])
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
EM_iter = 500 #30
tol = 10**(-9)
n_states = 3

A_mat = np.array([[0.6, 0.4, 0], [0.2, 0.7, 0.1],[ 0, 0.4, 0.6]])
#A_mat = None

pi = np.array([1, 0, 0])
#pi = None

drop_perc = 0.7
states1 = dont_drop(values = 2, states = states.copy(), drop_perc = drop_perc)
#states1 = None

e = 10**(-7)
label_mat = np.log( [[1-3*e,e,e,e], [1-3*e,e,e,e], [e,e,e, 1-3*e]] )
#labels_mat = None

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

testf = data2d_pdts.iloc[:,0: sel]
pl_ts =  data2d_pdts.iloc[:, sel:]
pl_ts['labels2'] = (pl_ts.labels >= 2).astype(int)
pl_ts['sum12'] = ((pl_ts.state1 + pl.state2) > threshold).astype(int)

# %% Printing
#take combinations of the labels
if label_mat is not None:
    print("With labeling matrix")

print("For Training Data")
print_summaries(data = pl, states = n_states)

print("\nFor Testing Data")

print_summaries(data = pl_ts, states = n_states)

# %% Supervised Training
C = 0.5
md1 = LogisticRegression(C = C)
md1 = md1.fit(trainf.values, pl.sum12.values)
prob_train = md1.predict_proba(trainf.values)[:,1].reshape([N,T])

md2 = LogisticRegression(C = C)
md2 = md2.fit(trainf.values, pl.labels2.values)



# %%evaluation
Nts = len(data_ts)
    
prob_test = md1.predict_proba(testf.values)[:,1].reshape([Nts,T])
prob_test2 = md2.predict_proba(testf.values)[:,1].reshape([Nts,T])
ytest = pl_ts.labels2.values.reshape([Nts, T])

lookback = 1
ev1, _ = time_series_multiple_th(prob_test, ytest, taus = 12, lookback = lookback)
                                
ev2, _ = time_series_multiple_th(prob_test2, ytest, taus = 12, lookback = lookback)


print(ev1['Auc'][0], ev2['Auc'][0])
pd.set_option('display.max_columns', 20)
display(ev1)
display(ev2)





# %%Plotting
mask = pl.labels2 == 1
x = np.random.uniform(low = -6, high = 6, size = 100000)

coef1 = md1.coef_[0]
int1 = md1.intercept_[0]
x21 = -x*coef1[0]/coef1[1] +(- int1  )/coef1[1]

coef2 = md2.coef_[0]
int2 = md2.intercept_[0]
x22 = -x*coef2[0]/coef2[1] + (-int2 )/coef2[1]



data2d = data.reshape([N*T, -1])
fig, ax = plt.subplots(1,1, figsize = (12,12))
ax.scatter(data2d[mask][:,0], data2d[mask][:,1], s = 0.1)
ax.scatter(data2d[~mask][:,0], data2d[~mask][:,1], s = 0.1)

ax.plot(x, x21)
ax.plot(x, x22)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(['Model 1', 'Baseline'])




# %% CLassification report
ypred1 = (prob_test >= 0.5).reshape(Nts*T)
ypred2 = (prob_test2 >= 0.5).reshape(Nts*T)

from sklearn.metrics import classification_report
print(classification_report(pl_ts.labels2.values, ypred1))
print(classification_report(pl_ts.labels2.values, ypred2))






