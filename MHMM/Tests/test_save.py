#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:30:56 2019

@author: george

test the save feature of hmms

"""

import sys
sys.path.append('../')
import  numpy as np
from _experiments import gauss_seq1d
from _misc import make_supervised, compute_forw, make_supervised2
from HMMs import MHMM
import matplotlib.pyplot as plt

np.random.seed( seed = 0)


mhmm = np.load('mymhmm.npy', allow_pickle = True).item()



mhmm2 = MHMM(MHMM_object = mhmm)
mhmm2 = mhmm2.fit( data = data, states = states1, dates = None, save_name = 'mymhmm.npy')

mhmm2par = mhmm2.get_params()