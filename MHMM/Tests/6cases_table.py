#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:57:53 2020
 
@author: george
"""
import numpy as np
from helper_functions.latex import latex_table




filename = 'tables/hmms.tex'
n_cols   = 7
caption  = "Absolute Error in the Recovery of the True HMM parameters."
titles   = ['Setting','Labels', 'Labeling Matrix', '$\\mc{A}_{err}$', '$\\mu_{err}$', '$\\Sigma_{err}$', '$\\pi_{err}$']
row_names = ['Unsupervised', 'Supervised', 'Few Labels Random', 'Few Labels Biased', 'Few Labels Biased']
row_numeric = np.load('tables/errors_mat.npy')[0:5].round(2)
row_text    = [['None', 'N/A'], ['Full', 'Correct'], ['4\\%', 'Correct'],
               ['4\\%', 'Correct'], ['4\\%', 'Incorrect']] 


row_values = [row_text, row_numeric]

label      = 'table:hmm_cases'


latex_table(filename, n_cols = n_cols, caption = caption, titles = titles,
            row_names = row_names, row_values = row_values, label = label)



 