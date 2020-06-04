#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 22:38:17 2020

@author: george
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl

def make_figures(X, Y, axis_names, fmts, labels, 
                 savename = None, style = 'seaborn-paper', fontsize = 20):
    
    """
    X: list with X axis of figures
    Y: list of Y axis of figures
    x_axis: name of x_axis
    y_axis: name of y_axis
    fmt: list of fromat strings for plots
    label: list of names for plots
    savename: name to save the plot
    
    """
    mpl.style.use(style)
    plt.rc('text', usetex = True)
    
    fig, ax = plt.subplots(1,1)
    
    for x,y, fm, lab in zip(X,Y, fmts, labels):
        ax.plot(x, y, fm, label = lab)
        
    
    ax.set_xlabel( axis_names[0], fontsize = fontsize )
    ax.set_ylabel( axis_names[1], fontsize = fontsize )
    ax.legend(prop={'size': fontsize})
    ax.tick_params(axis = 'both', labelsize = fontsize-2)
    plt.show()  
    
    if savename:
        fig.savefig(fname= savename +'.pdf', bbox_inches = 'tight')
    
    return ax

# %% start evaluations
evs = pickle.load(open('early_prediction_results/evals.p', 'rb'))
model     = evs[0].iloc[2:]
baseline  = evs[6].iloc[2:]

model1     = evs[0]
baseline1  = evs[6]

cols = evs[0].columns


# %% plots test
axis_names = [r'Threshold($\tau$)', 'True Positives per Hour']


labels = ['Relabeling', 'No Relabeling']
fmts   = ['d-k', 'o--k']
name   = 'tables/tp_synthetic'
X      = [baseline.tau, baseline.tau]
Y      = [model['tp/ts'], baseline['tp/ts']]
ax     = make_figures(X,Y, axis_names, fmts, labels, savename = name)

# %% plots2

axis_names = [r'Threshold($\tau$)', 'False Positives per hour']


labels = ['Relabeling', 'No Relabeling']
fmts   = ['d-k', 'o--k']
name   = 'tables/fp_synthetic'
X      = [baseline.tau, baseline.tau]
Y      = [model['fp/ts'], baseline['fp/ts']]
ax     = make_figures(X,Y, axis_names, fmts, labels, savename = name)


# %% plots3

axis_names = [r'Threshold($\tau$)', r'Average Delta']

labels = ['Relabeling', 'No Relabeling']
fmts   = ['d-k', 'o--k']
name   = 'tables/delta_synthetic'
X      = [baseline.tau, baseline.tau]
Y      = [model['avg_delta'], baseline['avg_delta']]
ax     = make_figures(X,Y, axis_names, fmts, labels, savename = name)

# %% plots4

axis_names = ['False Positive Rate', 'True Positive rate']

labels = ['Relabeling', 'No Relabeling']
fmts   = ['d-k', 'o--k']
name   = 'tables/roc_synthetic'
X      = [model1.t_fpr, baseline1.t_fpr]
Y      = [model1.t_recall, baseline1.t_recall]
ax     = make_figures(X,Y, axis_names, fmts, labels, savename = name)
# %%precision recall curves
fig, ax = plt.subplots(1,1)


ax.plot(model.tau, model['tp/ts'], label = 'Relabelling')
ax.plot(baseline.tau, baseline['tp/ts'], label = 'Baseline')
ax.legend()
ax.set_title('Tp per time series')


fig, ax = plt.subplots(1,1)
ax.plot(model.tau, model['fp/ts'], label = 'Relabelling')
ax.plot(baseline.tau, baseline['fp/ts'], label = 'Baseline')
ax.legend()
ax.set_title('Fp per time series')

fig, ax = plt.subplots(1,1)
ax.plot(model.tau, model['avg_delta'], label = 'Relabelling')
ax.plot(baseline.tau, baseline['avg_delta'], label = 'Baseline')
ax.legend()
ax.set_title('Average Delta')

fig, ax = plt.subplots(1,1)
ax.plot(model.t_fpr, model.t_recall, label = 'Relabelling')
ax.plot(baseline.t_fpr, baseline.t_recall, label = 'Baseline')
ax.legend()
ax.set_title('Precision Recall')
plt.subplots_adjust( hspace = 0.5, wspace = 0.5)



# %% matrix pick a threshold
t = 0.9
row_model = model[model.tau == t]
row_baseline = baseline[baseline.tau == t]







