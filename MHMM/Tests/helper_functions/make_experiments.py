#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:33:52 2020

@author: george
"""

import numpy as np
import pandas as pd




#specialized for three states
def get_experiment(X = None, y = None, option = 0, th = [0.5, 0.5, 0.5] ):
    
    """
    get the datasets for 3 state hmm experiments
    
    
    """
    X = X.copy()
    y = y.copy()
    
    if option == 1:
        
        Xo, yop =  option1(X, y, th)
        
        return Xo, yop
    
    if option == 2:
        
         Xo, yop =  option2(X, y, th)
        
         return Xo, yop
    
    
    if option == 3:
        
         Xo, yop =  option3(X, y, th)
        
         return Xo, yop
    
    if option == 4:
        
         Xo, yop =  option4(X, y, th)
        
         return Xo, yop
    
    if option == 5:
        
         Xo, yop =  option5(X, y, th)
        
         return Xo, yop
        
    
    
    
    
def option1(X, y, threshold):
     print('Options 1s: 0s versus Combination of 1s and 2s')
     
     [t0, t1, t2] = threshold
     
     
     #getting only the points satisfying the thresholds
     mask = (y.state0.values > t0) | (y.state1.values > t1) | (y.state2.values > t2)
     
     ynew = y[mask]
     xnew = X[mask]
     
     yfinal = (ynew['state0'] < t0).astype(int)
     
     return xnew, yfinal
 
    
def option2(X, y, threshold):
     
     print('Option 2: 0s versus  2s')
     [t0, t1, t2] = threshold
     
   
     
     mask = (y.state0.values > t0) | (y.state2.values > t2)
     
     ynew = y[mask]
     xnew = X[mask]
     
     
     yfinal = (ynew['state2'] >= ynew['state0']).astype(int)
     
     return xnew, yfinal
 
    
def option3(X, y, threshold):
     
     print('Option 3: 0s versus  1s')
     [t0, t1, t2] = threshold
     
     
     mask = (y.state0.values > t0) | (y.state1.values > t1)

     
     ynew = y[mask]
     xnew = X[mask]
     
     
     yfinal = (ynew['state1']>= ynew['state0']).astype(int)
     
     return xnew, yfinal
 
    
def option4(X, y, threshold):
     
     print('Option 4: 0s +  1s, versus 2s')
     [t0, t1, t2] = threshold
     
     
     mask = (y.state0.values > t0) | (y.state1.values > t1) | (y.state2.values > t2)
     
     ynew = y[mask]
     xnew = X[mask]
     
     
     yfinal = (ynew['state2'] > t2).astype(int)
     
     return xnew, yfinal
    
def option5(X, y, threshold):
     
     print('Option 5: 0s +  1s, versus 1s+2s')
     [t0, t1, t2] = threshold
     
     
     mask = (y.state0.values > t0) | (y.state1.values > t1) | (y.state2.values > t2)
     
     
     ynew = y[mask].reset_index()
     xnew = X[mask]
     
     st1            = np.where(ynew.state1.values > t1)[0]
     ynew['new']    = 0
     ynew.new[st1]  = (ynew.state2[st1].values > ynew.state0[st1].values).astype(int)
     
     yfinal = ynew['new'] + (ynew.state2 > t2).astype(int)
     
     
     
     
     yfinal = (ynew['state2'] > t2).astype(int)
     
     return xnew, yfinal
     
     
     
     
     
     
     
     
     
     
     