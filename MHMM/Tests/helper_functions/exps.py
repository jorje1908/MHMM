from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from HMMs import MHMM
import numpy as np
from series_proc import relabel, time_series_multiple_th, relabel2






def train_hmm(data = None, labels = None, hmm_labels = None, 
              label_mat = None, n_HMMS = 1, n_states = 3, EM_iter = 50, n_Comp = 1, tol = 10**(-4), 
              states_off = 0, A = None, pi = None,
              scale = True, Pca = 10, 
              tau_low = 0.5, tau_high = 0.7, state0 = 0, state1 = 1, comb = 1, hmm = None, releb = 2):
    
    """
    
    Helper Function Performing The relabelling and the HMM training
    
    data: Train data: (NxT)xD (2D)
    labels: Train Labels: (2D) NXT
    hmm_labels: hmm transformed labels (2D) NxT
    label_mat: Label Model For HMM
    n_HMMs:  Number of HMMs for training 
    n_states: Number of HMM states
    states_off: 0 or 1 (Deactiavet States or Not)
    A: Transition Matrix for HMM
    pi: State distributiom
    scale:  Scale the data or Not
    Pca: Number of PCA components
    tau_low: Value to label a state as Zero
    tau_high: Sum of other states to be labeled as 1
    state0: Which state to consider for the 0 label
    state1: which state to consider for label 1
    comb: if to combine the third state in one of the two 
            for relabeling
    
    
    return: 1) the trained hmm model
            2) the labels for the next stage of learning (1D)
            3) the 2d train data for training
            4) The pca model
            5) The scaler model
    
    
    """
    #CHECK SCALE
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        
        print("STEP 1: SCALING DATA")
    
    #CHECK PCA
    if Pca > 0:
        pca = PCA(n_components  = Pca)
        data = pca.fit_transform( data )
        
        print("STEP 2: PERFORMING PCA")
    else:
        pca = PCA()
    
    #NUMBER OF INSTANCES
    N = labels.shape[0] 
    
    #NUMBER OF TIME POINTS
    T = labels.shape[1]
    
    #NUMBER OF FEATURES
    D = data.shape[1]
    
    #3D DATA FOR HMM TRAINING
   
    #RESHAPE DATA TO BE 3D
    train = data.reshape([N,T,D])
    
    
    
    #HIDDEN MARKOV MODEL TRAINING
    if hmm is None:
        model = MHMM(n_HMMS = n_HMMS, n_states = n_states, EM_iter = EM_iter, n_Comp = n_Comp, tol = tol)
        model = model.fit(data = train, states = hmm_labels, states_off = states_off, label_mat = label_mat,
                                                                                          A = A, pi = pi)
   
        #GET THE PARTICULAR HMM
        hmm = model.HMMS[0]
    
    #REPORT
    unbiased = True
    if label_mat is None:
        unbiased = False
        
    semi = True
    if hmm_labels is None:
        semi = False
        
    sup_in = True
    if states_off == 0:
        sup_in = False
        
    
    
    
    #BEFORE RELABELING
    y_sum = np.sum(labels)
    print("\n \n")
    print("Number of States:{} \nGaussian Components:{} \nUnbiased: {} \nSemi Supervised: {}"\
          .format(n_states, n_Comp, unbiased, semi))
    
    print("Sup-Unsup Initialization: ", sup_in)
    print("Number of points before:{}".format(N*T))
    print("Number of Seizure Patients in Train Data: {}".format(np.sum(np.any(labels, axis = 1))))
    
    #Relabel
    if releb != 2:
        labels1d, probas_HMM, mask, indx = relabel(train, labels, hmm, tau_low = tau_low, 
                                                             tau_high = tau_high, state0 = state0, label_mat = label_mat, 
                                                              states = hmm_labels, state1 = state1, comb = comb)
        train2d = train.reshape([N*T, D])
        if len(mask) > 0:
        #print(mask.shape, train2d.shape)
        #print(mask)
            train2d = train2d[indx]
    
        print("Number of points after:{}".format(len(train2d)))
        
        return hmm, labels1d, train2d, probas_HMM, scaler, pca

        
    else:
       
        train2d = relabel2(train, hmm_labels, hmm, labels, label_mat = label_mat)
        
        return hmm, train2d, scaler, pca
    
    

    




