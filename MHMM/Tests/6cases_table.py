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
titles   = ['Cases','Labels', 'Labelling Matrix','$\\pi_{err}$', '$\\mc{A}_{err}$', '$\\mu_{err}$', '$\\Sigma_{err}$']
row_names = ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5']
row_numeric = np.load('tables/errors_mat.npy')[0:5].round(2)
row_text    = [['None', 'N/A'], ['Full', 'Correct'], ['Partial', 'Correct'],
               ['Partial', 'Correct'], ['Partial', 'Incorrect']] 


row_values = [row_text, row_numeric]

label      = 'table:hmm_cases'


latex_table(filename, n_cols = n_cols, caption = caption, titles = titles,
            row_names = row_names, row_values = row_values, label = label)



 