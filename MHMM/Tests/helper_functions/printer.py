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
    pl['argmax'] = np.argmax(pl[states1].values, axis = 1)
   
    
    print('Number of Positive Time Series:{} (state: {})'.format(positive_ts, states-1))
    print("Number of Initial Positive Labels: ", (pl.labels == 2).sum())
    
    #relabeled states
    for i, state in enumerate(states1):
        print("Hmm {}:{}".format(i,(pl[state] >= 0.5).astype(int).sum()))
    
    #ground truth
    for i, state in enumerate(states1):
        print("Ground Truth {}s: {} ".format(i, np.sum(pl.labels == i)))
    
    
    avg = None
    print("Relabeling Accuracy: ", accuracy_score(pl.labels.values, pl.argmax.values) )
    print("Relabeling Precision: ", precision_score(pl.labels.values, pl.argmax.values, average = avg) )
    print("Relabeling Recall: ", recall_score(pl.labels.values, pl.argmax.values, average = avg) )
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true = pl.labels, y_pred = pl.argmax, 
                           labels = [i for i  in range(len(states1))]))
    
    return
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    