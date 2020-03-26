#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:58:36 2020

@author: george
"""
# %%imports
import sys
sys.path.append('../')
sys.path.append('helper_functions/')
import  numpy as np
from _experiments import gauss_seq
from _misc import  compute_forw, dont_drop
from HMMs import MHMM
import matplotlib.pyplot as plt
from series_proc import relabel2, time_series_multiple_th
from sklearn.linear_model import LogisticRegression

###############################################################################
#Part 1: 
# %%Data Generation
np.random.seed( seed = 0)

#Initialize Transition Matrix
a0 = [0.8, 0.2, 0]
a1 = [0.2, 0.6, 0.2]
a2 = [0, 0.2, 0.8]

# a0 = [0.98, 0.02, 0]
# a1 = [0.29, 0.7, 0.01]
# a2 = [0, 0.4, 0.6]

#Initialize Means and Standard Deviations
mean = np.array([[-1,-1],[0,0], [1,1]])
std = np.array([[1,1],[1,1], [1,1]])
pi = [1,0,0]
A = np.array([a0, a1, a2])

#Initialize Time T and Number of Time Series N
T = 1000
N = 100

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
n_HMMS = 1
n_Comp = 1
EM_iter = 1000 #30
tol = 10**(-6)
n_states = 3
A_mat = np.array([[0.6, 0.4, 0], [0.2, 0.7, 0.1],[ 0, 0.4, 0.6]])
#A_mat = None
pi = np.array([1, 0, 0])
#pi = None
states1 = dont_drop(values = 2, states = states.copy(), drop_perc = 0.8)
#states1 = None

labels_mat = np.log( [[1,0,0,0], [1,0,0,0], [0,0,0,1]] )
#labels_mat = None
mhmm = MHMM(n_HMMS = n_HMMS, n_states = n_states, n_Comp = n_Comp, EM_iter = EM_iter, tol = tol)
mhmm = mhmm.fit( data = data, states = states1, dates = None, save_name = None, states_off = 0,
                label_mat = labels_mat, A = A_mat, pi  = pi)
#get the hmm
hmm = mhmm.HMMS[0]

#Get some Hmm Parameters
params = hmm.get_params()
cov = params['cov'].reshape(-1, mean.shape[1], mean.shape[1])

###############################################################################
# %%Relabelling
func = 'gmas'
data2d_pd = relabel2(data, hmm,  labels_hmm = states1, 
                     labels = states, label_mat = labels_mat, func = func)

data2d_pdts = relabel2(data_ts, hmm,  labels_hmm = None, 
                     labels = states_ts, label_mat = None, func = func)

sel = -5

trainf = data2d_pd.iloc[:,0:sel]
pl = data2d_pd.iloc[:, sel:]

testf = data2d_pdts.iloc[:,0: sel]
pl_ts = data2d_pdts.iloc[:, sel]

#take combinations of the labels
pl['sum12'] = ((pl.state1 + pl.state2) > 0.5).astype(int)
pl['sum02'] =  ((pl.state0 + pl.state2) > 0.5).astype(int)
pl['sum01'] =  ((pl.state0 + pl.state1) > 0.5).astype(int)
pl['only0'] = ((pl.state0 ) > 0.5).astype(int)
pl['only1'] = ((pl.state1 ) > 0.5).astype(int)
pl['only2'] = ((pl.state2 ) > 0.5).astype(int)
pl['argmax'] = np.argmax(pl[['state0', 'state1', 'state2']].values, axis = 1)
pl['labels2'] = (pl.labels >= 2).astype(int)

pl_ts['sum12'] = ((pl_ts.state1 + pl_ts.state2) > 0.5).astype(int)
pl_ts['sum02'] =  ((pl_ts.state0 + pl_ts.state2) > 0.5).astype(int)
pl_ts['sum01'] =  ((pl_ts.state0 + pl_ts.state1) > 0.5).astype(int)
pl_ts['only0'] = ((pl_ts.state0 ) >= 0.5).astype(int)
pl_ts['only1'] = ((pl_ts.state1 ) >= 0.5).astype(int)
pl_ts['only2'] = ((pl_ts.state2 ) >= 0.5).astype(int)
pl_ts['argmax'] = np.argmax(pl_ts[['state0', 'state1', 'state2']].values, axis = 1)
pl_ts['labels2'] = (pl_ts.labels >= 2).astype(int)

print("For Training Data")
print("Number of Initial Labels: ", (pl.labels == 2).sum())
print("States 0 + 1: ", pl.sum01.sum())
print("States 0 + 2: ", pl.sum02.sum())
print("States 1 + 2: ", pl.sum12.sum())
print("States 0: ", pl.only0.sum())
print("States 1: ", pl.only1.sum())
print("States 2: ", pl.only2.sum())
print("Real 0s: ", np.sum(pl.labels == 0))
print("Real 1s: ", np.sum(pl.labels == 1))
print("Real 2s: ", np.sum(pl.labels == 2))
print("Relabeling Accuracy: ", np.sum(pl.argmax == pl.labels)/len(pl) )


print("\n\nFor Testing Data")
print("Number of Initial Labels: ", pl_ts.labels.sum())
print("States 0 + 1: ", pl_ts.sum01.sum())
print("States 0 + 2: ", pl_ts.sum02.sum())
print("States 1 + 2: ", pl_ts.sum12.sum())
print("States 0: ", pl_ts.only0.sum())
print("States 1: ", pl_ts.only1.sum())
print("States 2: ", pl_ts.only2.sum())
print("Real 0s: ", np.sum(pl_ts.labels == 0))
print("Real 1s: ", np.sum(pl_ts.labels == 1))
print("Real 2s: ", np.sum(pl_ts.labels == 2))
print("Relabeling Accuracy: ", np.sum(pl_ts.argmax == pl_ts.labels)/len(pl_ts) )

# %% Supervised Training

md1 = LogisticRegression()
md1 = md1.fit(trainf.values, pl.sum12.values)

md2 = LogisticRegression()
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

#display(ev1)
#display(ev2)
















data2d = data.reshape([N*T, -1])
plt.scatter(data2d[:,0], data2d[:,1], s = 0.1)









