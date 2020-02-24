import numpy as np
import pandas as pd
from scipy.integrate import trapz
import time

def time_series_analysis(probabilities, labels, dates = None, tau = 0.5, lookback = 0, return_pos = 0, Dminus = 0):
    """
    Function to process time series data:
    
    probabilities : matrix-like (NxD) N number of instances, 
    D number of time steps, each entry is a probability in [0-1]
    
    labels: matrix-like (NxD), labels matrix 0 or 1 for every 
    entry of the "probabilities" matrix
    
    dates: Nx2 matrix having starting index date and ending index date
    
    tau: probability threshold to calculate the metrics: default 0.5
    return_pos = to return positions of TPs TNs, FPs, FNs
    
    returns: Average Delta (time difference between positive prediction 
                            and true state (0 or 1))
             Number of True Positives
             Number of True Negatives
             Number of False Positives
             Number of False Negatives
    """
    
    #initialize counters to 0
    T = probabilities.shape[1]
    TP = 0
    TPlist = []
    
    TN = 0
    TNlist = []
    
    FP = 0
    FPlist = []
    
    FN = 0
    FNlist = []
    
    De = 0
    total_D = 0 
    Dlist = []
    
    if probabilities.ndim == 0:
        print(" Wrong input")
        return
    elif probabilities.ndim == 1:
        print(" Two dimensional Matrix needed, if it is a vector \
              reshape it to have dimensions (N,1)")
    else:
        N = probabilities.shape[0]
        D = probabilities.shape[1]
        
    #Threshold probabilities based on tau
    proba = probabilities >= tau
    proba = proba.astype(float)
    
   # check = []
    #start the analysis
    for i in range(N):
       # print(i)
        if dates is None:
            st = 0
            ed = T-1
        else:
            st = dates[i,0]
            ed = dates[i,1]
        
        proba_lookback = prob_lookback(proba[i,:], st, ed, lookback)
        delta = compute_delta(proba_lookback, labels[i,st:ed+1])
        
        if delta != np.inf:
            De += delta
            #print(D, delta)
            Dlist.append([delta, np.max(probabilities[i, st:ed +1])])
            total_D += 1
            
            if Dminus == 1 and delta < 0:
                TP -= 1
                FN += 1
                De -= delta
                total_D -= 1
                
        
        proba_i = proba_lookback
        labels_i = labels[i,st:ed+1]
        pr_pos = len(np.where(proba_i == 1)[0])
        tr_pos = len(np.where(labels_i == 1)[0])
        
        #check.append([pr_pos,tr_pos])
        if pr_pos >= 1 and tr_pos >= 1:
            TP += 1
            TPlist.append(i)
            
        if pr_pos == 0 and tr_pos >= 1:
            FN += 1
            FNlist.append([i, np.argmax(probabilities[i, st:ed +1])+st])
            
        if pr_pos >= 1 and tr_pos == 0:
            FP += 1
            FPlist.append([i, np.argmax(probabilities[i, st:ed +1])+st])
            
        if pr_pos == 0 and tr_pos == 0:
            TN += 1
            TNlist.append(i)
        
    
    if total_D == 0:
        delta_av = np.inf    
    else:
        delta_av = De/total_D
        
    accuracy = (TP + TN)/N
    recall = (TP)/(TP+FN)
    specificity = FP/(FP+TN) if (FP + TN) > 0 else -np.inf
    try:
        precision = TP/(TP+FP)
    except:
        precision = 0
        
    names = ['Tau','Average D', 'TP','FP','TN', 'FN','Accuracy', 'TPR', 'FPR', 'Precision','#TS']
    results = [tau,delta_av, TP, FP, TN, FN, accuracy, recall, specificity, precision, N ]
    results_pd = pd.DataFrame(data = np.array(results).reshape(1,11), columns = names)
    
    if return_pos == 0:
        return results_pd, Dlist #, check
    
    return results_pd, Dlist, [ np.array(TPlist), np.array(TNlist), np.array(FPlist), np.array(FNlist) ]
    
def compute_delta( probas, labels):
    """
    
    Takes a timeseries with probabilities
    and the corresponding labels and finds the distance
    of the predicted 1 to the real first 1
    
    delta can be
    positive: (first referal after first predicted referal)
    negative: (first real referal before first predicted referal)
    zero: referal at the same point
    
    """
    
    ones_probas = np.where(probas == 1)[0]
    ones_labels = np.where(labels == 1)[0]
    
    if len(ones_probas) == 0 or len(ones_labels) == 0:
        return np.inf
    
    first_one_in_probas = ones_probas[0]
    first_one_in_labels = ones_labels[0]
    
    delta = first_one_in_labels - first_one_in_probas
    
    return delta


def prob_lookback(probability, st, ed, lookback):
    """
    
    takes a vector of probability already 
    thresholded by a tau and converts it
    to a new 0-1 vector taking 
    in consideration the lookback
    
    """
    proba_new = np.zeros_like(probability)
    
    counter = 0
    for i in range(st, ed + 1):
        
        pi = probability[i]
        #check if we have an 1 in position i
        if pi == 1:
            counter += 1 #if we have an 1  increase the counter
            
        if pi == 0:
            counter = 0
            
        if counter > lookback : # if counter bigger than lookback
            proba_new[i] = 1
     
    return proba_new
        
        
        

def proba_hmm( hmm, data3d, prob = 1):
    """
    returns a matrix with probabilities
    calculated by the HMM
    
    hmm: an hmm class object
    data3d: matrix with data points DxTxF
    
    prob: calculates the probability of being
    at state "prob" 0 or 1: default 1
    
    """
    
    N = data3d.shape[0]
    T = data3d.shape[1]
    
    proba_mat = np.zeros( shape = [N,T])
    
    for i in range(N):
        point = data3d[i]
        proba =  np.exp(hmm.log_forward(point))[int(prob), :]
        
        proba_mat[i,:] = proba
        
    return proba_mat

def hmm_proba_3d_ref(data3d, dates, ref, hmm, tau = 0.5, value = 1):
    """
    
    assigns labels in the training points for the 
    SGMM training based on hmm
    
    
    """
    ref1 = ref.copy()
    N = data3d.shape[0]
    probas = []
    for i in range(N):
        st = int(dates[i, 0])
        end = int(dates[i, 1])
        
        point = data3d[i, st:end+1,:]
        proba = np.exp(hmm.log_forward(point))[value, :]
        probas.append(proba)
        for j in range(st, end+1):
            if proba[j-st] >= tau:
                ref1[i,j] = 1
                
    return ref1, probas

def hmm_proba_3d_ref2(data3d, dates, ref, hmm, tau = 0.5, tau_low = 0.3, value = 1, value_low = -1):
    """
    
    assigns labels in the training points for the 
    SGMM training based on hmm
    
    
    """
    ref1 = ref.copy()
    N = data3d.shape[0]
    probas = []
    for i in range(N):
        st = int(dates[i, 0])
        end = int(dates[i, 1])
        
        point = data3d[i, st:end+1,:]
        proba = np.exp(hmm.log_forward(point))[value, :]
        probas.append(proba)
        for j in range(st, end+1):
            if proba[j-st] >= tau:
                ref1[i,j] = 1
                
            if proba[j-st] <= tau_low:
                if ref1[i,j] != 1:
                    ref1[i,j] = value_low
                
    return ref1, probas

def relabel(data3d, labels, hmm, tau_low = 0.5, tau_high = 0.7, state0 = 0,
            state1 = 1, label_mat = None, states = None, comb = 0):
    """
    relabeling
    data
    
    """
    N = data3d.shape[0]
    T = data3d.shape[1]
    labels = labels.copy()
    K = hmm.states_
    probas = np.array([]).reshape([0,T])

    third_state = np.setdiff1d(np.arange(K), np.array([state0, state1]))[0]
    print("third state: ", third_state) 
    print("Labels before relabelling: {}".format(np.sum(labels)))
    
    for i in range(N):
        point = data3d[i,:,:]
        prob = np.exp(hmm.log_forward(point, label_mat = label_mat, states = states[i]))
        probas = np.concatenate([probas,prob], axis = 0)
        
        #proba_state = prob[state]
        for j in range(T):
            if comb == 0:
                if prob[state0][j] >= tau_low and (labels[i,j] != 1):
                    labels[i,j] = -1
                
                if (prob[state1][j]) >= tau_high:
                    labels[i,j] = 1
            elif comb == 1:
                if (prob[state0][j] + prob[third_state][j]) >= tau_low and (labels[i,j] != 1):
                    labels[i,j] = -1
                
                if (prob[state1][j]) >= tau_high:
                    labels[i,j] = 1
                    
            elif comb == 2:
                if (prob[state0][j] ) >= tau_low and (labels[i,j] != 1):
                    labels[i,j] = -1
                
                if (prob[state1][j] + prob[third_state][j]) >= tau_high:
                    labels[i,j] = 1
            else:
                if (prob[state0][j] ) >= tau_low :
                    labels[i,j] = -1
                
                if (prob[state1][j]) >= tau_high:
                    labels[i,j] = 1
                
                
    probas = np.array(probas).reshape([hmm.states_, N*T]).T           
    labels1d = labels.reshape([N*T,1])
    mask = labels1d == 0
    indx = np.where(labels1d != 0)[0]
    if len(mask > 0):
        labels1d = labels1d[~mask]
        probas = probas[indx]
    
    labels1d[labels1d == -1] = 0
    print("Labels after relabelling: {}".format(np.sum(labels1d)))
    
    return labels1d, probas, mask, indx


def relabel2(data3d, hmm,  labels_hmm = None, labels = None,
             label_mat = None, func = 'forw'):
    
    """
        data3d: 3D data NxTXD
        labels_hmm : NxT labels with -infs inside
        hmm: hmm model
        labels: real hmm labels
        label_mat: labeling matrix
     
    """
    
    N, T, D = data3d.shape
    features = ['feat{}'.format(i) for i in range(1, D+1)]
    K = hmm.states_
    states_proba = np.zeros(shape = [N,T, K])
    
    for i in range(N):
        st = None if labels_hmm is None else labels_hmm[i]
        
        if func != 'gamas':
            states_proba[i] = np.exp(hmm.log_forward(X = data3d[i], states = st, label_mat = label_mat)).T
            
        else:
            
            states_proba[i] = np.exp(hmm.log_gamas(X = data3d[i])).T
            
    

    states_proba_2d = states_proba.reshape([N*T, K])
    data_tran_pd = pd.DataFrame( data3d.reshape([N*T,D]), columns = features )
    
    for k in range(K):
        data_tran_pd[['state'+str(k)]] = pd.DataFrame(states_proba_2d[:,k], 
                                                    index = data_tran_pd.index)
    
    data_tran_pd['labels'] = labels.reshape([N*T])
    
    if labels_hmm is not None:
        data_tran_pd['labels_hmm'] = labels_hmm.reshape([N*T])
        
    else:
        data_tran_pd['labels_hmm'] = data_tran_pd['labels']
    
    return data_tran_pd
                
def probas_hmm(data3d, dates, hmm, value = 1):
    """
    takes a 3d matrix and the dates
    of the data and the calculates
    the probabilities of the hmm
    
    """
    
    N = data3d.shape[0]
    T = 60
    
    proba_mat = np.zeros(shape = [N,T])
    
    for i in range(N):
        st = int(dates[i, 0])
        end = int(dates[i, 1])
        
        point = data3d[i, st:end+1,:]
        proba = np.exp(hmm.log_forward(point))[value, :]
        proba_mat[i, st:end+1] = proba[:]
                
    return proba_mat


def time_series_multiple_th(probabilities, labels, dates = None, taus = 10, lookback = 0):
    """
    evaluates the time series analysis for multiple taus
    uniformly distributed in the taus specified
    
    """
    
    tau_n = np.linspace(start = 0.00, stop = 1, num = taus, endpoint = False)
    resu0, dlist = time_series_analysis(probabilities, labels, dates, tau = tau_n[0])
    resu0 = resu0.values
    Dlists = []
    Dlists.append(dlist)
    for i in np.arange(1, len(tau_n) ):
        
        resu, dlist = time_series_analysis(probabilities, labels, dates, tau = tau_n[i], lookback = lookback )
        resu = resu.values
        resu0 = np.concatenate((resu0, resu), axis = 0)
        Dlists.append(dlist)
    
    names = ['Tau','Avg D', 'TP','FP','TN', 'FN','accuracy', 'TPR', 'FPR', 'Precision', '#TS']
    results_pd = pd.DataFrame(data = np.array(resu0), columns = names)
    results_pd['Auc'] = -trapz(results_pd['TPR'].values, results_pd['FPR'].values)
    
    return results_pd.round(3), Dlists
        
def survival_evaluation (probabilities = None, labels = None, enhanced_labels = None, dates = None, tau = 0.5, forward = 6, backward = 0):
    """
    probabilities: PatientsxT pandas matrix
    labels: pandas matrix PatientsxT the corresponding labels
    enhanced_labels: expanded matrix of labels for faster evaluation
    dates: Nx2 with dates patients are in the data
    tau: threshold to apply to probabilities
    forward: forward look horizon
    backward: backard look horizon
    
    """
    
    if enhanced_labels is None:
        enhanced_labels  = expand_references(labels, forward, backward)
    
    #thresholded probabilities
    thresh_prob = probabilities.copy()
    thresh_prob[thresh_prob >=tau] = 1
    thresh_prob[thresh_prob < tau] = 0
   
    
    #Evaluate for each time step
    Pat = labels.shape[0] #number of patients
    N = 0 #total points at each time step 
    T = probabilities.shape[1]
    brier_count = pd.DataFrame(np.zeros(len(labels)), index = labels.index)
    ev = np.zeros(8) #evaluations matrix
    D = 0
    D_total = 0
    
    #dates diff
    if dates is not None:
        dat_diff = dates.iloc[:,1]-dates.iloc[:,0] +1
    else:
        dat_diff = labels.shape[1]
    
    for t in range(T): #for every time t
        if t % 100 == 0:
            print(t)
        
        #mask that takes all the points in a specific time step
        #according to the inclusion dates
        if dates is not None:
            mask = (dates.iloc[:,0] <=t ) & (dates.iloc[:,1] >= t) 
        else:
            mask = np.ones(shape = [Pat], dtype = bool)
        
        #take the points of the thersholded values
        thresh_prob_t = thresh_prob[mask].iloc[:,t]
        
        #take the points of the probabilities
        probabilities_t = probabilities[mask].iloc[:,t]
        
        #take the enhanced labels
        enhanced_labels_t = enhanced_labels[mask].iloc[:,t]
        
        #take the real labels accounting only for the specified horizon
        labels_mask = labels[mask].iloc[:, t: t+ forward +1]
        
        #compute metrics per time
        eval_metrics, brier, D_part, D_tot = survival_evaluation_t( thresh_prob_t, probabilities_t, 
                                                                   enhanced_labels_t, labels, time = t )
        
        #add the evaluatons
        ev += eval_metrics
        
        #aggragate brier
        brier_count[mask] += brier
        
        #increase Delta
        D += D_part
        D_total += D_tot
        N += len(thresh_prob_t)
    
    #final delta
    D = D/D_total
    
    #True positives
    TP = ev[0]
    
    #False Positives
    FP = ev[1]
    
    #True Negatives
    TN = ev[2]
    
    #False negatives
    FN = ev[3]
    
    #time averaged accuracy
    acc = ev[4]/T
    
    #time averaged precision
    prec = ev[5]/T
    
    #time averaged recall
    tpr = ev[6]/T
    fpr = ev[7]/T
    
    #total accuracy
    accuracy = (TP+TN)/N
    
    #total precision
    precision = TP/(TP + FP) if (TP+FP) > 0 else 0
    
    #total recall
    TPR = TP/(TP+FN) if (TP+FN) > 0 else 0
    FPR = FP/(FP+TN) if (FP+TN) > 0 else 0
    #average brier score
    brier_av = np.mean(brier/dat_diff)
    
    #pack everything in a pandas matrix to return them
    names = ['TAU','DELTA', 'BRIER','TP','FP','TN', 'FN','ACC', 'TPR', 'FPR', 'PREC', 'ACC_T', 'PREC_T', 'TPR_T','FPR_T','Tot_P']
    results = [tau, D, brier_av, TP, FP, TN, FN, acc, tpr, fpr, prec, accuracy, precision, TPR,FPR, Pat]
    
    #panda matrix to be returned
    results_pd = pd.DataFrame(data = np.array(results).reshape(1,len(results)), columns = names)
    
    return results_pd
    
def survival_evaluation_t(thrPr_t, proba_t, enh_lab_t, labels = None, time = -1):
    """
    
    evaluation for each time step t
    this can be run from survival_evaluation function
    
    """
    D_part = 0
    #number of points at time t
    N = len(thrPr_t) 
    
    #mask for true positives
    maskTP = (thrPr_t == 1) & (enh_lab_t == 1) 
    
    #taking all the ids with TP from labels
    labels_TP = labels[maskTP]
    
    #number to divide Delta
    D_tot = np.sum(maskTP)
    
    #compute partial delta
    if D_tot >= 1:
        #print(time)
        D_part = compute_delta_surv(labels_TP.copy(), time)
       
    
    #True positives
    TP =  np.sum(maskTP)
    
    #False Positives
    FP =  np.sum((thrPr_t == 1) & (enh_lab_t == 0))
    
    #True Negatives
    TN = np.sum((thrPr_t == 0) & (enh_lab_t == 0))
    
    #False Negatives
    FN = np.sum((thrPr_t == 0) & (enh_lab_t == 1))
    
    #accuracy
    accuracy = (TP+TN)/N
    
    #precision
    precision = TP/(TP + FP) if (TP+FP) > 0 else 0
    
    #recall
    TPR = TP/(TP+FN) if (TP+FN) > 0 else 0
    FPR = FP/(FP+TN) if (FP+TN) > 0 else 0
    
    
    #brier score
    brier_score_t = (enh_lab_t - proba_t)**2
    
    return np.array([TP,FP,TN,FN, accuracy, precision, TPR, FPR]), brier_score_t, D_part, D_tot

def row1(x):
    """
    takes a dataframe row and
    finds the position of the first 1
    """
    #xx = x.values
    
    
    i = np.where(x == 1)[0]
    if len(i) > 0:
       # print(i[0])
        return i[0]
    else:
        return -1
    
def compute_delta_surv(labels, t):
    """
    computes total Delta for a specific time
    uses row1 function for apply in pandas
    
    """
    I = labels.apply(func = row1 , axis = 1)
    D = np.sum(I - t)
   
    return D

def survival_evaluation_multiple (probabilities = None, labels = None, enhanced_labels = None, dates = None, taus = [0,0.2,0.4,0.6,0.8,0.9], forward = 6, backward = 0):
    
    """
    Calls survival evaluation for multiple therholds
    """
    
    if enhanced_labels is None:
        enhanced_labels = expand_references(labels.copy(), forward, backward)
        
    for i, tau in enumerate(taus):
        if i == 0:
            res = survival_evaluation(probabilities = probabilities, 
                                  labels = labels, enhanced_labels = enhanced_labels, dates = dates, tau = tau, forward = forward, backward = backward)
        else:
            res2 = survival_evaluation(probabilities = probabilities, 
                                  labels = labels, enhanced_labels = enhanced_labels, dates = dates, tau = tau, forward = forward, backward = backward)
            res =  pd.concat([res, res2], ignore_index=True)
    
    return res

def expand_references(data, forward, backward):
    """
    expands the labels of reference data
    expands the 1's in data forward values
    in the back and backward values forward
    
    data: pandas array
    forward: values to copy 1's backwards
    backward: values to copy 1's forward
    
    """
    start = time.time()
    #get the data rows bigger with sum bigge than 0
    #only those we need to proceed
    data_pos = data[data.sum(axis = 1) > 0]
    
    #loop through data_pos and expand labels
    print("Positive Patients: ",len(data_pos))
    for i in range(len(data_pos)):
        x_i = data_pos.iloc[i,:]
        indx_pos = np.where(x_i == 1)[0]
        
        for j in indx_pos:
            #check bounds backward
            if  j-forward < 0:
                data_pos.iloc[i,0:j] = 1
                
            else:
                
                data_pos.iloc[i, j-forward:j  ] = 1
                
            #check forward bounds   
            if j+backward >= len(x_i):
                data_pos.iloc[i,j:] = 1
            else:
                data_pos.iloc[i,j: j+backward + 1] = 1
                
    data[data.sum(axis = 1) > 0] = data_pos
    
    end = time.time() - start
    print("expand references took {}s".format(end))
    return data

                