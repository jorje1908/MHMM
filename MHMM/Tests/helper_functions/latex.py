#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:06:19 2020

@author: george
"""

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