#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:06:19 2020

@author: george
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl


def latex_table(filename = None, n_cols = -1, caption = None, titles = [], row_names = [],
     row_values = [], math = True, label = None):
    f = open(filename, 'w+', n_cols)
    
    """ 
    filename: name of the file to save
    n_cols: How many columns the file wil have
    caption: the caption of the file
    titles: the first row titles
    row_names: the names of the first coluumn rows
    row_values:  the values of the rows after the first row, these are split
     in text and math row_values = [list_text, list_math]
    math = True, puts the math symbol on the row values
    
    
    """
    if filename is None:
        print('no file name given exiting the program')
        return 0
    #first column left format all later columns center
    form = 'l '
    cols = 'c '*(n_cols-1)
    form += cols
    lines = []
    caption = caption
    
    #first row
    columns = titles
    lines.append('\\begin{table}[!htb]\n')
    lines.append('   \\centering \n')
    lines.append('   \\caption{'+ caption + '}' +'\n')
    lines.append('   \\label{'+label+'}'+'\n')
    lines.append('   \\begin{tabular}{'+form+' }\n')
    lines.append('     \\toprule \n')
    
    title = "".join('     '+col+'& ' for col in columns[:-1]) + columns[-1]+'\\\ \n'
    lines.append(title)
    lines.append('     \\midrule \n')
    
    #body of the file
    if math:
        
        for i, row in enumerate(row_names):
             ss = ''
             ss +='     \\textbf{'+row_names[i]+'} '
             row_text = row_values[0][i]
             ss  = ss + "".join('&'+value+' ' for value in row_text)
             row_vals = row_values[1][i]
             ss = ss + "".join('&$'+str(value)+'$  ' for value in row_vals)
             ss = ss + '\\\ \n'
             lines.append(ss)
    
    lines.append('     \\bottomrule \n \n')
    lines.append('   \\end{tabular} \n')
    lines.append('\\end{table}')
    
    f.writelines(lines)
    print('table saved succesfully')
    f.close()
    
    
    
    
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