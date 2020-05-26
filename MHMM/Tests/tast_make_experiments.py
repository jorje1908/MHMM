#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:15:30 2020

@author: george
"""

import numpy as np
import pandas as pd


np.random.seed(seed = 0)

from helper_functions.make_experiments import get_experiment

yv = np.random.rand(10,3)
yv = yv/np.sum(yv, axis = 1)[:,None]

X, y = np.random.rand(10,3), pd.DataFrame(yv, 
                                          columns = ['state0','state1','state2'])





th    = [0.2, 0.4, 0]
Xn,yn = get_experiment(X = X, y = y, option = 5, th = th)