#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:21:57 2020

@author: george
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


np.random.seed(seed = 34567)


inputs = {'n_samples': 15000, 'n_features':2, 'n_redundant':0,'flip_y':0.01,
          'random_state': 80, 'weights':[0.5]}

X, y = make_classification(**inputs)






plt.figure()
X0 = X.copy()
N0 = len(X0)
C = 1
C0 = C/N0
lg0 = LogisticRegression(penalty = 'l2', tol = 10**(-17), C = C0, random_state = 5252)
lg0.fit(X0,y)
print('LG0')
print(np.mean(X0, axis = 0), np.var(X0, axis = 0))
print(lg0.coef_, lg0.intercept_)
b0  = lg0.intercept_[0]
w10 = lg0.coef_[0,0]
w20 = lg0.coef_[0,1]

proba0 = lg0.predict_proba(X0)[:,1]

ylin0 = -np.arange(-2,2, 0.02)*w10/w20 -b0/w20
plt.scatter(X0[y==1][:,0], X0[y==1][:,1], s= 1, label = '1')
plt.scatter(X0[y==0][:,0], X0[y==0][:,1],s = 1, label = '0')
plt.plot(-np.arange(-2,2, 0.02), ylin0)

plt.show()


X1 = X.copy()
N1 = len(X1)
C1 =C*N1

lg1 = LogisticRegression(penalty = 'l2', tol = 10**(-17), C = C1, random_state = 5252)
X1[:,0] = X1[:,0]*1000 + 2000
lg1.fit(X1,y)
proba1 = lg1.predict_proba(X1)[:,1]

print('LG1')
print(np.mean(X1, axis = 0), np.var(X1, axis = 0))
print(lg1.coef_, lg1.intercept_)
b1  = lg1.intercept_[0]
w11 = lg1.coef_[0,0]
w21 = lg1.coef_[0,1]

t1 = np.arange(10,80, 0.02)
ylin1 = -t1*w11/w21 -b1/w21
plt.figure()
plt.scatter(X1[y==1][:,0], X1[y==1][:,1], s= 1, label = '1')
plt.scatter(X1[y==0][:,0], X1[y==0][:,1],s = 1, label = '0')
plt.plot(t1, ylin1)


plt.show()




X2 = X.copy()
X2 = np.concatenate([X2,X2,X2,X2,X2,X2,X2,X2,X2,X2], axis = 0)
y2 = np.concatenate([y,y,y,y,y,y,y,y,y,y])

N2 = len(X2)
C2 =C*N2

lg2 = LogisticRegression(penalty = 'l2', tol = 10**(-17), C = C2, random_state = 5252)
X2[:,0] = X2[:,0]*1000 + 2000
lg2.fit(X2,y2)
proba2 = lg2.predict_proba(X2)[:,1]

print('LG2')
print(np.mean(X2, axis = 0), np.var(X2, axis = 0))
print(lg2.coef_, lg2.intercept_)
b2  = lg2.intercept_[0]
w12 = lg2.coef_[0,0]
w22 = lg2.coef_[0,1]

t2 = np.arange(10,80, 0.02)
ylin2 = -t2*w12/w22 -b2/w22
plt.figure()
plt.scatter(X2[y2==1][:,0], X2[y2==1][:,1], s= 1, label = '1')
plt.scatter(X2[y2==0][:,0], X2[y2==0][:,1],s = 1, label = '0')
plt.plot(t2, ylin2)

proba_all = np.array([proba0, proba1])


