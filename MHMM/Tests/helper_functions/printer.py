#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:57:58 2020

@author: george

Summary Module
main Purpose to Print summaries 
of the Relabeled Matrices Created after evaluating the trained HMM in the data
the labeling function used 
is under the mhmm project

Directory:
MHMM/Tests/helper_functions/series_proc.py 

function Name relabel2




"""

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
#import seaborn as sns 
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score




def print_summaries(data = pd.DataFrame(), states = -1, states2d = None):
    """
    

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is pd.DataFrame().

    states = Number of HMM states
    Returns
    -------
    None.
    
    using the  Output dataframe produced by the relabel function
    as described above,
    it prints some summaries of the data
    
    the data are of the form : features --> feat1, feat2, ...., featD
                             : states probabilities --> state0, state1, ...
                             : Real labels --> labels
                             : Hmm Labels (labels given to HMM for Training)
                              -->labels_hmm

    """
    if states == -1:
        print("Error: You need to give the number of HMM states")
        return
    
    #Number of Positive Time Series
    positive_ts = (states2d == states-1).any(axis = 1).sum()
    states1 = ['state'+str(i) for i in range(states)]
    pl = data.copy()
    sum12 = ((pl.state1 + pl.state2) > 0.5).astype(int)
    sum02 =  ((pl.state0 + pl.state2) > 0.5).astype(int)
    sum01 =  ((pl.state0 + pl.state1) > 0.5).astype(int)
    only0 = ((pl.state0 ) > 0.5).astype(int)
    only1 = ((pl.state1 ) > 0.5).astype(int)
    only2 = ((pl.state2 ) > 0.5).astype(int)
    pl['argmax'] = np.argmax(pl[states1].values, axis = 1)
    pl['labels2'] = (pl.labels >= 2).astype(int)
    
    print('Number of Positive Time Series:{} (state: {})'.format(positive_ts, states-1))
    print("Number of Initial Positive Labels: ", (pl.labels == 2).sum())
    print("HMM states 0 + 1: ", sum01.sum())
    print("HMM states 0 + 2: ", sum02.sum())
    print("HMM states 1 + 2: ", sum12.sum())
    print("Hmm 0: ", only0.sum())
    print("Hmm 1: ", only1.sum())
    print("Hmm 2: ", only2.sum())
    print("Ground Truth 0s: ", np.sum(pl.labels == 0))
    print("Ground Truth 1s: ", np.sum(pl.labels == 1))
    print("Ground Truth 2s: ", np.sum(pl.labels == 2))
    
    avg = None
    print("Relabeling Accuracy: ", accuracy_score(pl.labels.values, pl.argmax.values) )
    print("Relabeling Precision: ", precision_score(pl.labels.values, pl.argmax.values, average = avg) )
    print("Relabeling Recall: ", recall_score(pl.labels.values, pl.argmax.values, average = avg) )
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true = pl.labels, y_pred = pl.argmax, 
                           labels = [i for i  in range(len(states1))]))
    
    return
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    