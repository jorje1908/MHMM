#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:50:20 2019

@author: george
"""

import sys
sys.path.append('../data/sales')
sys.path.append('..')
#sys.path.append('/home/george/github/sparx/code/data/sales')
import numpy as np
import pandas as pd
import time
from _utils import generate_Coin


from HMMs import MHMM
from HMMs_1 import MHMM as MH
from hmmlearn import hmm
from _visu import plot_prob_state
np.random.seed( seed = 2 )

sales = pd.read_csv("/home/george/github/sparx/code/data/sales/sales.csv")
salesNorm = sales.iloc[:, 55:].values
salesNorm3d = np.expand_dims( salesNorm, axis = 2)

N = 100
data, states, coins = generate_Coin(N = N)

#initialize MHMM
model = MHMM(n_HMMS = 1, n_states = 2, EM_iter = 400, tol = 10**(-12), n_Comp = 1)

states_sup = np.full( shape = [1,N], fill_value = -np.inf)
#states_sup[0,5] = states[5]
#states_sup[0,63] = states[63]
#fit HMM
start = time.time()
model = model.fit( data = data[1:] , states = None)
end = time.time() - start



print("time elapsed: {:.4}".format(end))
logLi = model.logLikehood
params = model.get_params()['params']

hmmM = model.HMMS[0]
gamas = hmmM.log_gamas( data[1] )
forw = hmmM.log_forward( data[1])
gamma_states = np.argmax(gamas, axis = 0)
vit_log, vit_states, vit_seq, p_states = hmmM.log_viterbi( data[1])

accuracy_viterbi =  np.sum( vit_seq == states[:,0])/N
accuracy_gamma = np.sum( gamma_states == states[:,0])/N
#start = time.time()

#m1 = m1.fit(data = data[1:] )
#end = time.time() - start
#print("time elapsed: {:.4}".format(end))
#
#m2 = m2.fit( data[1])
#logLi1 = m1.logLikehood
#params1 = m1.get_params()['params']   

"""
if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', sort='time')
    
"""