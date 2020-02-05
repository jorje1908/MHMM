#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:55:28 2019

@author: george

custom K means with
Kmeans ++ 

"""

import numpy as np



def custom_kmeans( data =  None, labels = None, K = None, itera = 100, tol = 10**(-3)):
    
    """
    Kmeans for initialization of MHMM
    it uses kmeans ++ for initialization
    and it fixes the known means into super points
    
    """
    
    #data dimension
    D = data.shape[1]
    
    #find unique labels
    unique = np.unique( labels )
    

    means = np.zeros( shape = [K,D] )
    
    #initialize super steady clusters
    st_means, st_lengths = _init_steady_means(data, labels, unique, D)
    
    #take points of consideration
    ind_inf = np.where( np.isinf(labels))[0]

    #initialize rest of the means with kmeans plus plus
    means = kmeanspp(data[ind_inf], means, st_means )
    
    dat_sq = np.sum( data[ind_inf]**2, axis = 1)
   # print('Main Body', means.shape, st_means.shape, 'Dat Sq', dat_sq.shape)
    
    data_inf = data[ind_inf]
    
    #create an index matrix to trace where the indexes go in terms of
    #the initial matrix
    index_mat = np.zeros(shape = [len(data_inf),2])
    index_mat[:, 0] = np.arange(len(index_mat))
    index_mat[:, 1] = ind_inf


    Jst = 10**25
    for i in range(itera):
        
        means, J, indexes = updates_means(data_inf, means, st_means,
                                     st_lengths, dat_sq, index_mat)
        
        print(np.abs(Jst-J), J)
        if np.abs(Jst-J) <= tol:
            break
        
        Jst = J
    
    
    indexes_new, means_o = _reformat_indexes(labels, K, indexes, unique, means)
    return means, indexes_new, J, st_means, means_o
    
    
    

def _init_steady_means(data, labels, unique, D):
    
    """
    based on the labels and the unique values 
    initializes the fixed means to be used in the algorithm
    
    """
    
    st_means = np.zeros( shape = [ len(unique)-1, D] )
    st_lengths = np.zeros(len(unique)-1)
    for i in range(1, len(unique)):
        
        ind = np.where( labels == unique[i] )[0]
        st_means[i-1] = np.mean( data[ind] )
        st_lengths[i-1] = len(ind)
        
    return st_means, st_lengths
    
 
def kmeanspp(data, means, steady_means):
    """
    kmeans plus plus initialization
    """
    
    
    K  = len(means)
    L = len(steady_means)
    N = len(data)
    
    #print('Kmeans ++', means.shape)
    for i in range(L):
        means[i] = steady_means[i].copy()
       
    
    
    for j in range(L, K):
        
        #find the distances between the current means
        #and all the points
        
        #means jxD , data NxD
        #terms -2ximi
        second_term = -2*means[0:j,:]@(data.T) #jxN
        means_term = np.sum(means[0:j,:]**2, axis = 1) #jx()
        data_term = np.sum(data**2, axis = 1) #Nx()
        
        almost_ready = second_term + data_term #jxN
        final_term = almost_ready.T + means_term   #NxJ
        
       # print("finla_term", final_term.shape)
        #Now final _term has the squared distances  of all the points 
        #from the first j means
        
        min_mat = np.min(final_term, axis = 1)
        
        min_norm = min_mat/(np.sum(min_mat))
        
        index_choose = np.random.choice(np.arange(N), size = 1, p = min_norm)[0]
        means[j,:] = data[index_choose]
    
    #print('Kmeans ++ 2', means.shape)
    return means
        
        
        
def updates_means(X, means, st_means, st_lengths, Xsq, index_mat = None):

    """
    one iteration of kmeans
    updates means for kmeans algorithm
    
    """  
    indexes = []
   # print("Update means", means.shape)
    L = len(st_means)     
        
    double_term = -2*(means@X.T) + Xsq #KxN
   # print("Update  means double term", double_term.shape)

    means_sq = np.sum(means**2, axis = 1)
    
    #print("Update  means means_sq", means_sq.shape)

    
    sq_dist = double_term.T + means_sq
    
   # print("Update  means sq_dist", sq_dist.shape)
    
    classes = np.argmin( sq_dist, axis = 1 )
    J = np.sum(np.min(sq_dist, axis = 1))
    
   # print("Update means Classes", classes.shape)

    
    #reestimate means
    for i in range(len(means)):
        
        if i < L:
            indx = np.where(classes == i)[0]
            sum_points = np.sum(X[indx], axis = 0)
            means[i,:] = (st_means[i,:]*st_lengths[i] + sum_points)/(st_lengths[i]+ len(indx))
            
            indexes.append(index_mat[indx,1])
        
        else:
            
            indx = np.where(classes == i)[0]
            means[i,:] = np.mean(X[indx], axis = 0)
            indexes.append(index_mat[indx,1])

            
            
    #print("Update means Means", means.shape)

    return means, J, indexes
                                                                         
        
#helper for the HMM initialization        
        
def _reformat_indexes(data, K, indexes, unique, means):

    """
    Reformating the indexes to be in the correct order
    for the HMM initialization
    
    """
    
    means_o = np.zeros_like(means)
    
    c = 0
    indexes_extended = []
    
    ordered_list = np.arange( K )
    reduced_list = np.delete( ordered_list, unique[1:] )
    final_list = np.concatenate((unique[1:], reduced_list))
    ff_order = np.zeros(K)
    for i in range(len(final_list)):
        
        ff_order[i] = np.where( final_list == i)[0]
        
    print('OL RL FL FFO',ordered_list, reduced_list, final_list, ff_order)

    
    for i in range(1, len(unique)):
        
        indx = np.where(data == unique[i])[0]
        indx2 = indexes[c] 
        indx_f = np.concatenate((indx, indx2))
        indexes_extended.append(indx_f)
        c += 1
        
    for i in range(len(unique)-1, K):
        indexes_extended.append(indexes[i])
        
        
    indexes_final = []
    
    for i in range(len(ff_order)):
        
        indexes_final.append(indexes_extended[int(ff_order[i])].astype(int))
        means_o[i,:] = means[int(ff_order[i]), :]
            
    return indexes_final, means_o
        
        
        
        
    
    
    
    