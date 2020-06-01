import time

import numpy as np
import pandas as pd
from numba import jit

#from time_series_evaluations_helper import expand_labels

###############################################################################################################################
#CODE FOR COST - TREATMENT BASED EVALUATION


@jit(nopython = True, cache = True, nogil = True, parallel = True)
def treatment_cost(J = 100, c = 10, H = 3, dates = np.array([[]]),
                   predictions = np.array([[]]), labels = np.array([[]])):
    
    
    """
    Average Cost of Multiple Time Series
    
    J : Out-of-Treatment False Negative Cost
    c :  Treatment Cost
    H : Treatment Horizon
    dates: Nx2 Two dimensional Numpy array with start and end values of time series
    predictions: NxT Two dimensional numpy array with 0-1 predictions for each time step
    labels: NxT Two dimensional numpy array with ground truth labels
    pt: print time default True
    
    """
    
    
    if predictions.ndim != 2 or labels.ndim !=2 or dates.ndim != 2:
        print("The input Arrays Need To be 2 Dimensional Numpy Arrays, Exiting the Program")
        return
    
    #number of time series
    N, T = predictions.shape
        
    total_cost = 0
    
    #loop over the time series
    for i in range(N):
        if N > 1:
            predictions_i = predictions[i]
            labels_i = labels[i]
        else:
            predictions_i = predictions[0,:]
            labels_i = labels[0,:]
        
        #get the start and end of the ith time series
        if dates.shape[1] > 1:
            start = dates[i,0] 
            end   = dates[i,1]
        else:
            start = -1 
            end = -1
        
        #get tthe cost for the ith time series
        cost = treatment_cost_ts(J = J, c = c, H =H, start = start, end = end, 
                                 predictions = predictions_i, labels = labels_i)
        
        total_cost += cost
    
    #return the averaged out cost
    total_cost = total_cost/N
    

    return total_cost

@jit(nopython = True, cache = True, nogil = True)
def treatment_cost_ts(J = 100, c = 10, H = 3, start = -1, end = -1,
                   predictions = np.array([]), labels = np.array([])):

    """
    The total cost of a time series
    
    J : Out-of-Treatment False Negative Cost
    c :  Treatment Cost
    H : Treatment Horizon
    start: starting position of time series
    end : ending position of time series
    predictions: 0-1 predictions for each time step
    labels: grund truth labels
    
    pt: print time default False
    """
   
    
    #if start and end not provided 
    #take the whole time series
    if start == -1 and end == -1:
        T = len(predictions)
        start, end = 0, T-1
        
    #start  processing
    #all tretaments are relative to the positive prediction from time 0 to time H-1
    treatment_stage = 0 
    treatment_active = False  #See if Treatment if Active or Not

    #initialize the cost for a time series
    total_cost = 0

    for t in range(start, end+1, 1):
        pred_t = predictions[t]
        label_t = labels[t]

        #Cases: In Treatment, not in Treatment

        #1: In treatment
        if treatment_active:
            #cases: We have event or Not
            if label_t == 1:
                #we are in treatment so we have a reduced cost
                total_cost += (c/H)*treatment_stage

                #reset treatment stage
                if pred_t == 1:
                    treatment_stage = 0
                    treatment_active = True 
            else:
                #Pay for the extra treatment
                if pred_t == 1:
                    total_cost += (c/H)*treatment_stage
                    treatment_stage = 0
                    treatment_active = True


            #increase treatment_stage by 1 time step
            treatment_stage += 1

            #when we reach the horizon value we get out
            #of Treatment
            if treatment_stage >= H:
                treatment_stage = 0
                treatment_active = False

        #2: Out of Treatment
        else:
            #cases: Event of Not, Prediction Or Not
            if label_t == 1:
                if pred_t == 1:
                    total_cost += c
                    treatment_stage = 1  #setting for the next time step
                    treatment_active = True

                else: #event with no prediction
                    total_cost += J

            else:

                if pred_t == 1:
                    total_cost += c
                    treatment_active = True
                    treatment_stage = 1
                    
    
        
    return total_cost


###############################################################################################################################
#CODE FOR SECOND TIME SERIES EVALUATION

@jit(nopython = True, cache = True, nogil = True, parallel = True)
def aggregate_evaluations(predictions = np.array([[]]), labels = np.array([[]]), 
                         forward_horizon = 3, backward_horizon = 0, 
                         dates = np.array([[]]), metrics = np.array([[]]) ):
    
    """
    predictions:      NxT numpy array with the label prediction in each time step
    labels:           NxT numpy array with the corresponding ground truth labels
    forward_horizon:  (int) Forward time window for the evaluation
    backward_horizon: (int) Backward time window for the evaluation
    dates:            (Nx2) numpy indicating the start and end of each time series
    metrics:          (Nx7) array passed from the shell because non python mode
                      do not support numpy initializations
    
    
    
    
    """
    
    
    if predictions.ndim != 2 or labels.ndim !=2 or dates.ndim != 2:
        print("The input Arrays Need To be 2 Dimensional Numpy Arrays, Exiting the Program")
        return
    
    #number of time series
    N, T = predictions.shape
    fw, bw = forward_horizon, backward_horizon
        
    #initialize Metrics Array (tp,tn,fp,fn, delta,ts_type,length)
    #metrics = np.zeros([N, 7])
    
    #loop over the time series
    for i in range(N):
        if N > 1:
            predictions_i = predictions[i]
            labels_i = labels[i]
        else:
            predictions_i = predictions[0,:]
            labels_i = labels[0,:]
        
        #get the start and end of the ith time series
        if dates.shape[1] > 1:
            start = dates[i,0] 
            end   = dates[i,1]
        else:
            start = -1 
            end = -1
            
        tp, tn, fp, fn, delta, ts_type, l = evaluate_time_series(predictions = predictions_i, labels = labels_i, 
                         forward_horizon = fw, backward_horizon = bw, 
                         start = start, end =  end)
        
        metrics[i,0] = tp
        metrics[i,1] = tn
        metrics[i,2] = fp
        metrics[i,3] = fn
        metrics[i,4] = delta
        metrics[i,5] = ts_type
        metrics[i,6] = l
        
        
    return metrics

def aggregate_evaluations_pd(predictions = np.array([[]]), labels = np.array([[]]), 
                         forward_horizon = 3, backward_horizon = 0, 
                         dates = np.array([[]])):
    
    """
    Wrapper of "aggregate_evaluations" function into a pandas array
    
    predictions:      NxT numpy array with the label prediction in each time step
    labels:           NxT numpy array with the corresponding ground truth labels
    forward_horizon:  (int) Forward time window for the evaluation
    backward_horizon: (int) Backward time window for the evaluation
    dates:            (Nx2) numpy indicating the start and end of each time series
    
    
    """
    N, T = predictions.shape
    mets = np.zeros([N,7])
    metrics = aggregate_evaluations(predictions = predictions, labels = labels,
                                    forward_horizon = forward_horizon, 
                                    backward_horizon = backward_horizon, dates = dates, metrics = mets)
    
    column_names = ['tp', 'tn', 'fp', 'fn', 'delta','ground_truth', 'length']
    
    
    metrics_pd = pd.DataFrame( metrics, columns = column_names)
    metrics_pd['forward_horizon'] = forward_horizon
    metrics_pd['backward_horizon'] = backward_horizon
    
    return metrics_pd

def summary_evaluations_pd(probabilities = np.array([[]]), labels = np.array([[]]), 
                         forward_horizon = 3, backward_horizon = 0, 
                         dates = np.array([[]]), tau_n = 3, tau_list = None,
                         ts_metrics = False):
    
    """
    Wrapper Function that Aggregates Statistics for all Time Series
    the only thing added are the thersholds (tau_n)
    to evaluate 
    """
    
    
    fh = forward_horizon
    bh = backward_horizon
    
    if tau_list is None:
        taus = np.linspace(0, 1, tau_n)
    else:
        taus = tau_list
    
    #loop through taus
    for i, tau in enumerate(taus):
    
        predictions = (probabilities.copy() >= tau).astype(int)
        mets_pd = aggregate_evaluations_pd(predictions = predictions,
                                       labels = labels,
                                       forward_horizon = fh, 
                                       backward_horizon = bh, dates = dates)
        
        
        
        if i == 0:
            results = process_evaluations_pd(mets_pd, tau)
        
        else:
            res = process_evaluations_pd(mets_pd, tau)
            results = pd.concat([results, res], axis = 0, ignore_index=True)
            
    if ts_metrics:
        return results, mets_pd
                                        
    return results
        

        


def process_evaluations_pd(metrics_pd, tau):
    
    
    """
    Takes the pandas Matrix returned From "aggregate_evaluations_pd"
    and computes a summary of this matrix
    
    
    return a summarized pandas matrix
    """
    
    #aggregated metrics
    agg = metrics_pd.copy()
    agg_mets = agg[['tp','tn', 'fp', 'fn','delta']].sum()
    fw = agg.forward_horizon.iloc[0]
    bw = agg.backward_horizon.iloc[0]
    
    
    N = len(agg)
    ts_accuracy = (agg_mets['tp'] + agg_mets['tn'])/(agg_mets['tp'] + 
                                                     agg_mets['fn']+agg_mets['tn'] 
                                                     + agg_mets['fp'])
    try:
        ts_recall =  agg_mets['tp'] /(agg_mets['tp'] + agg_mets['fn'])
    except:
        ts_recall = 0
        print("Division By 0 will set to 0")

    try:
        ts_fpr    = agg_mets['fp'] /(agg_mets['fp'] + agg_mets['tn'])
    except:
        ts_fpr = 0
    
    try:
        ts_precision =  agg_mets['tp'] /(agg_mets['tp'] + agg_mets['fp'])
    except:
        ts_precision = 0
        print("Division By 0 will set to 0")

        
    avg_mets = agg_mets/N
   
    avg_length = agg['length'].mean()
    
    
    help_neg = agg[agg['ground_truth'] < 1]
    help_pos = agg[agg['ground_truth']>0]
    
    neg = len(help_neg)
    tn = neg-len(help_neg[help_neg['fp'] > 0])
    fp = neg - tn

    pos = len(help_pos)
    tp  =  len(help_pos[help_pos['tp'] >= 1])
    fn  = pos - tp
    
    avg_mets['delta'] = (avg_mets['delta']*N)/pos
   

    agg_accuracy = (tp+tn)/(tn+tp+fn+fp)
    
    try:
        agg_recall = (tp)/(tp+fn)
    except:
        agg_recall = 0
        print("Division By 0 will set to 0")
        
    try:
        agg_precision = (tp)/(tp+fp)
    except:
        agg_precision = 0
        print("Division By 0 will set to 0")

        
    try:
        fpr = fp/(fp+tn)
    except:
        fpr = 0
        print("Division By 0 will set to 0")

    
    columns = ['tau','ts_length', 'fw', 'bw', 'tp/ts', 
             'tn/ts','fp/ts', 'fn/ts','avg_delta', 't_accuracy', 
              't_precision', 't_recall', 't_fpr', 'tp','tn', 'fp', 'fn',
              'agg_accuracy', 'agg_precision', 'agg_recall','fpr','pos','neg', 'total_ts']
    
    mets = np.zeros([1, len(columns)])
    mets[0,0] = tau
    mets[0,1] = int(avg_length)
    mets[0,2:4] = [int(fw),int(bw)]
    mets[0,4:9] = avg_mets.values
    mets[0, 9] = ts_accuracy
    mets[0,10] = ts_precision
    mets[0,11] = ts_recall
    mets[0,12] = ts_fpr
    mets[0, 13: 17] = np.array([tp, tn, fp, fn]).astype(int)
    mets[0, 17: 21] = [agg_accuracy, agg_precision, agg_recall, fpr ]
    mets[0, 21:24] = [int(pos), int(neg), int(N)]
    
    mets_pd = pd.DataFrame(mets, columns = columns).round(4)
    
    
    return mets_pd


@jit(nopython = True, cache = True, nogil = True, parallel = True)
def evaluate_time_series(predictions = np.array([]), labels = np.array([]), 
                         forward_horizon = 3, backward_horizon = 0, 
                         start = -1, end = -1 ):
    
    """
    Evalauate A time Series, On the Following Metrics:
    Tp = True Positives:
    Tn = True Negatives
    Fp = False Positives
    Fn = False Negatives
    Delta: Time From first prediction to the event:
    
    
    predictions = (T,) nd Array of the predictions for a time series
    labels = (T,) correspodning ground truth labels
    forward_horizon(window): window within the prediction holds true in the future
    backward_prediction(window): window within the prediction holds true in the past
    start: time the time series start (for time series not starting at 0)
    end: time the time series end (for time series not ending at T)
    pt: print running time(default 0)
    
    returns: tp,tn,fp.fn, delta, ts_type
    ts_type: 1 if there is an one in the ground truth labels
             0 else
             
    expanded labels for check
    
    """
    
    
    
    #condense the names into new variables
    fh = forward_horizon 
    bh = backward_horizon 
    
    #expand labels according to the forward and backward horizon
    labels, position = expand_labels(labels = labels.copy(), forward_horizon = fh, backward_horizon = bh)
    
    #initialize the relevant variables
    tp,tn,fp,fn, delta = 0,0,0,0, 0
    
    #first Time we see a Positive Prediction
    #we turn the gate to False
    gate = True
    
    #time series type positive: 1, negative: 0
    ts_type = 1
    
    #it mean this is a negative time series(there is no 1 in the labels)
    if position == -1:
        gate = False
        ts_type = 0
        
    if start == -1 and end == -1:
        start = 0
        end = len(predictions) - 1
    
    #calculate time series length
    length = end - start + 1
        
        
    #for loop to calculate values
    for i in range(start, end+1, 1):
        pred_i = predictions[i]
        label_i = labels[i]
        
        #if our prediction is potive 
        #we can have either tp or fp
        if pred_i == 1:
            
            #find delta
            if  gate:
                delta = position - i 
                gate = False
            
            if label_i == 1:
                tp += 1
            else:
                fp += 1
        #if our prediction is negative
        #we can have either tn or fn
        else:
            
            if label_i == 1:
                fn += 1
            else:
                tn += 1
    
        
    return tp, tn, fp, fn, delta, ts_type, length 



###############################################################################################################################
# HELPER FUNCTIONS

@jit(nopython = True, cache = True, nogil = True)
def expand_labels(labels = np.array([]), forward_horizon = 3, backward_horizon = 0):
    """
    
    modify the ground truth labels for faster processing 
    of the window
    labels: Numpy 1 Dimensional Array with binary labels 0 and 1
    forward_horizon(window): how much to look forward in the future
    backward_horizon(window): how much backward to see in the future
    
    returns: modified labels, (maybe the position of the first labels for Delta)
    """
    #Length of the Time Sequence
    T = len(labels)
    fh = forward_horizon
    bh = backward_horizon
    first_positive = -1
    
    #guard for first positive position
    gate = True
    
    #forward pass
    for i in range(T):
        #take the ith label
        li = labels[i]
        
        #check the value
        if li == 1:
            
            if gate:
                first_positive = i
                gate = False
            
            #expand the labels
            #future expanding
            if i>= fh:
                labels[i-fh:i] = 1
                
            else:
                labels[0:i] = 1
                
    #if we define a backward horizon
    #bigger than 0
    if bh > 0:
        
        #reverse the labels
        reverse = labels[::-1].copy()
        for i in range(T): 
            #take the ith label
            li = reverse[i]

            #check the value
            if li == 1:

                #expand the labels
                #future expanding
                if i>= bh:
                    reverse[i-bh:i] = 1

                else:
                    reverse[0:i] = 1
                    
        unreverse = reverse[::-1]
        #labels = np.round((labels+unreverse)/2)
        for j in range(T):
            if unreverse[j] == 1:
                labels[i] = 1
    

    
    return labels, first_positive




