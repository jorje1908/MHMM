#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:08:17 2019

@author: george

Implements 
--> A hidden Markov Model and extends to
--> a Mixture of Hidden Markov Models,

--> Observational Model is consisted by a Mixture of 
Gaussians Model,

--> The Model will be trained by the Expectation Maximization
    Algorithm 


"""

import numpy as np
from numpy import logaddexp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
#from MHMM import _hmmh
import time
import _utils



#########     check functions   #########
def checkShape(  arg1, arg2, name):
        
    if arg1.shape != arg2.shape:
        print( "Warning shapes does not match in " + name)
        
        
    return

def checkSum_one( matrix, axis, name):
    """
    Checks if the matrix entries along the given axis
    sum to 1
    """
    
    result = matrix.sum( axis = axis ).round(5)
    value = np.all( result == 1 )
    
    
    if not value:
        print(" Warning: Elements do not sum to 1 in {} ".format(name))
        
    return

def checkSum_zero( matrix, axis, name):
    """
    Checks if the matrix entries along the given axis
    sum to 1
    """
    
    result = logsumexp(matrix, axis = axis ).round(5)
    value = np.all( result == 0 )
    
    
    if not value:
        print(" Warning: Elements do not sum to 0 in {} ".format(name))
        
    return

class HMM(object):
    """  
    HMM class 
    
    implements all the basic HMM algorithms
    in addition to an extension to be used by a 
    Mixture of Hidden Markov Models class MHMM
    
    
     Attributes:
        
        t_cov -->  type of covarinace matrix for gaussian components
                   "full" or "diag"
        states_ --> Number of States of the Hidden Markov model
        
        g_components_ --> number of gaussian components by state
        
        A --> state transition matrix
        
        pi --> initial state probabilities
        
        cov --> covarinace matrices
        
        means --> gaussian means
        
        gauss --> gaussian components by state by component
        
        alpha --> prior probabilities of gaussian components
        
        gmm_init --> initialization of the gaussians "Kmeans" or "Custom"
        
        kmeans_points --> how many points to use for the initialization
        
        initialized --> if the HMM has been initialized
    
    
    """
    
    
    
    def __init__(self, states = 2, g_components = 2, t_cov = "diag", 
                 gmm_init = 'Kmeans', kmean_Points = -1, idi = None):
        
        #setting covariance type attribute
        self.t_cov = t_cov
        #Setting states attribute
        self.states_ = states
        #Setting Gaussian components attribute
        self.g_components_ = g_components
        #initialize Transition Matrix
        self.A = np.zeros( shape = [states, states], dtype = 'float')
        #initilize initial distribution probabilities np.array pi[k] k = 1...K
        self.pi = np.zeros( shape = states, dtype = 'float')
        
        #4d array cov[k,l,d,d] kth state lth gaussian component
        self.cov = None
    
        #3d array means[k,l,d] kth state lth mean
        self.means = None
    
        #list of gaussian predictors gauss[k,l] lth gaussian of kth state
        #each gauss[k][l] component is an instance of the multivariate_normal
        #scipy object
        self.gauss = None
        #initial probabilities of each gaussian component, for each state
        #alpha[k,l] kth state lth component
        self.alpha = None
        #how to initialize Mixture of Gaussians 
        self.gmm_init = gmm_init
        #how many random points to use for the kmeans
        self.kmean_Points = kmean_Points
        #state of the model initialized or not -1 means all
        self.initialized = False
        #iD of the HMM
        self.id = idi
        
        self.timeIn_p_States = 0 #myTEst
        
        
    #ALGORITHMS OF HMM
    
        
    def log_predict_states_All(self, X ):
        """
        computes the probability to observe a x_t for all t = 1...T
        for all states K = 1...K
        for any observation in time it calculates the probability 
        using the method predict_states
        
        X  = (T,d) is a sequence of observations x1, ... xT
        numpy array Txd
        
        return the matrix P[k,t] probability to observe x_t given state k
        
        p(x_t | z_t = k, H = m) for all t for all k
        
        P = [K,T] matrix
        
        """
        
        #get the number of sequneces in time
        T = X.shape[0]
        #get number of states
        K = self.states_
        #initialize the array
        log_P_all = np.zeros( shape = [K, T])
        
        for t in np.arange( T ): #for every time step
           log_P_all[:, t] = self.log_predict_states( X[t] )
        
        return log_P_all
        
    def log_predict_states(self, x):
        """ 
         calculates the probability of x to be observed given a state,
         for all states and returns a matrix 
         P(x) = [p(x_t|z_t = 1)... p(x_t|z_t = K)]
         
         x = (d, )
         
         retuns P(x) = (K,)
        
        """
        #take the number of states
        K = self.states_
        #initialize P
        log_P = np.zeros( shape = [K] )
        
        for k in np.arange( K ): #for every state 
            log_P[k] = self.log_predict_state( x, st = k)
         
       
        return log_P
        
        
    def log_predict_state(self, x, st = None):
         """
         calculates the probabiity of an observation to be generated 
         by the state "state" --> p(x_t | z_t = k)
         
         returns a matrix kx1
         x = (d,)
         
         """
         
         #get number of Gaussian Components
         components = self.g_components_
         #initialize probability to 0
         log_pk = -np.math.inf #was 0
         
         for component in np.arange( components ): #for every G compo
             log_pk = logaddexp(log_pk,
                    self.log_predict_state_comp( x, st = st, cmp = component))
          
        
         return log_pk
            
         
         
    def log_predict_state_comp(self, x, st = None, cmp = None):
         
         """ 
         predicts the probability of an observation to be generated
         by the lth component gaussian of the kth state
         
         st: HMM  state {1...K}
         cmp: Kth state lth Gaussian Component l = {1...L}
         
         p( G_t = l, x_t |  z_t = k) = 
         p(x_t | G_t = l, z_t = k)p(G_t = l|z_t = k)
         
         x = (d,)
         
         """
         
         #get the multivariate_gaussian of state = "state"
         #component = "comp"
         gaussian_kl = self.gauss[st][ cmp ]
         #get the probability of this component to be chosen
         log_alpha_kl = np.log(self.alpha[st, cmp])
         #predict the probability of x to be generated by the lth component
         #of
         pkl = gaussian_kl.logpdf( x ) + log_alpha_kl
         
         return pkl
     
    #forward algorithmm 
    def log_forward(self, X, log_p_states = None, states =  None, post = False):
        """
        Implements the forward Algorithm, 
        Returns tha matrix forw [ a_1,... a_T ]
        where a_t = [at[1], ... ai[K]]^(transpose)
        where ai[k] = p( z_i = k, x_1...x_t)
        
        X  = (T,d ) is a sequence of observations x1, ... xT
        numpy array Txd

        
        """
       
        #get length of obseravtions in time
        T = X.shape[0]   
        
        #get number of states
        K = self.states_
        
        #get transition matrix Aij = p(z(t) = j|z(t-1) = i)
        log_A = np.log(self.A)
        
        #initialize forward matrix
        log_forw = np.zeros( shape = [K, T], dtype = np.double)
        
        #take initial alphas a_1
        if log_p_states is None:
            log_p_states = self.log_predict_states_All( X )
         
        #use Cython to speed up forward
        """
        _hmmh._forward_log( log_A,  log_p_states,
              np.log(self.pi),  log_forw,
              T,  K)
        """
        _utils._log_forward( log_A,  log_p_states,
              np.log(self.pi),  log_forw,
              T,  K, states = states)
        
        if post:
            posterior = _utils._log_forward( log_A,  log_p_states,
              np.log(self.pi),  log_forw,
              T,  K, states = states)
            
            return log_forw, posterior
        
        
        return log_forw
    
    def log_viterbi(self, X, log_p_states = None):
        """
        Implements the viterbi algorithm
        X  = (T,d ) is a sequence of observations x1, ... xT
        numpy array Txd
        """
        #get length of obseravtions in time
        T = X.shape[0]   
        
        #get number of states
        K = self.states_
        
        #get transition matrix Aij = p(z(t) = j|z(t-1) = i)
        log_A = np.log(self.A)
        
        #initialize forward matrix
        log_vit = np.zeros( shape = [K, T], dtype = np.double)
        
        #take initial alphas a_1
        if log_p_states is None:
            log_p_states = self.log_predict_states_All( X )
         
        #use Cython to speed up forward
        """
        _hmmh._forward_log( log_A,  log_p_states,
              np.log(self.pi),  log_forw,
              T,  K)
        """
        log_vit, states, seq = _utils._log_viterbi( log_A,  log_p_states,
              np.log(self.pi),  log_vit,
              T,  K)
        
        
        return log_vit, states, seq, log_p_states
    
    def log_predict_x( self, X , states = None):
        """
        X = (T,d )is a Txd matrix T observations of d dimension
        predict the probability of X to be generated by the model
        log p( x | H = m) = log p(x_1,...,x_T | H = m)
        
        it does this by calling the forward algorithm
        and takes the sum of probabilities of last time step
        
        """
        #take the time 
        T =  X.shape[0] 
        
        #compute forward probabilities
        log_forw, posterior = self.log_forward( X, states = states, post = True )
        
        #sum of aj(T) j = 1 ... K
        log_pX = logsumexp( log_forw[:, T-1] )
        
        return posterior #log_pX
    
    def log_backward(self, X, log_p_states = None):
        """
        Implements the Backward algorithm
        Returns the matrix backw [b_1,..., b_T]
        where b_t = [bt[1],..., bt[K]]
        where bt[k] = p(x_(t+1)...x_T| z_t = k)
        
        
        X =(T,d) is a sequence of observations x1, ... xT
        numpy array Txd
        
        backw = [K,T]
        
        """
        
        #get length of obseravtions in time
        T = X.shape[0]   
        
        #get number of states
        K = self.states_
        
        #get transition matrix Aij = p(z(t) = j|z(t-1) = i)
        log_A = np.log(self.A)
        
        #initialize backward matrix
        log_backw = np.zeros( shape = [K, T], dtype = np.double)
        
        if log_p_states is None:
            log_p_states = self.log_predict_states_All(X)
        
        
        #attempt to speed up with Cython
        """
        _hmmh._backward_log(log_A,  log_p_states,
                      np.log(self.pi), log_backw,
                      T,  K)
        """
        _utils._log_backward(log_A,  log_p_states,
                             log_backw,
                              T,  K)
        
        return log_backw
    
    #smoothing probabilities
    def log_gamas(self, X, log_forw = None, log_backw = None):
        
        """
        Computes the probability of being at state k at time t given
        we have observed all the sequence x1...xT,
        it is called  smoothed probability on hmms 
        p(z_t = k| O1...OT)
        
        X = (T,d) is a sequence of observations x1, ... xT
        numpy array Txd
        
        returns the gamma matrix  a KxT numpy array K are the HMM states
        T is the length of sequence in time
        
        """
        
        #regularization
        reg = 10**( -6 ) #used if needed
        
        #run forward algorithm_
        if log_forw is None:
            log_forw = self.log_forward( X )
            
        #run backward algorithm
        if log_backw is None:
            log_backw = self.log_backward( X )
        
        K = self.states_
        log_gamma = np.zeros( shape = [K,K])
        
        log_gamma = _utils._log_gamas(log_forw, log_backw, log_gamma)
        
#        #calculate gamma unnormalized
#        log_gamma = log_forw + log_backw 
#       
#        #normilize gamma
#        log_gamma = log_gamma - logsumexp(log_gamma, axis = 0)

        
        #print("Checking gamma")
        checkSum_zero( log_gamma, axis = 0 , name = 'log_gamas')
       
        return log_gamma

    
    def sliced( self, X, log_forw = None, log_backw = None, log_p_states = None):

        """
        Computes the sliced probability  p(z_t = i, Z_(t+1) = j| o1...T)
        
        returns a KxKxT-1 matrix  xis with the aformentioned probabilities
        
        X = (T, d)
        """
        
        #get length of obseravtions in time
        T = X.shape[0]   
        
        #get number of states
        K = self.states_
        
        #get transition matrix Aij = p(z(t) = j|z(t-1) = i)
        log_A = np.log(self.A)
        
        #initialize xis matrix
        log_xis = np.zeros( shape = [K, K, T-1 ] )
        
        #compute observation proabilities for all time steps of X
        if log_p_states is None:
            log_p_states = self.log_predict_states_All( X )
            
        #compute forward
        if log_forw is None:
            log_forw = self.log_forward( X )
            
        #compute backward 
        if log_backw is None:
            log_backw = self.log_backward( X )
        """ 
        _hmmh._xis_log(log_A, log_p_states, log_forw,
                       log_backw, log_xis, T,K)
        """
        _utils._log_xis(log_A, log_p_states, log_forw,
                       log_backw, log_xis, T,K)
       
       
        #print("Checking xis")
        checkSum_zero( logsumexp(log_xis, axis = 0), axis = 0, name = 'log_xis' )
        #print(logsumexp(logsumexp( log_xis, axis = 0), axis = 0))
        return log_xis
    
    #Gaussian Component Posterior
    def log_g(self, x_i):
        """
        calculates the posterior probabilities 
        
        p(G = l|x_t, H = m, z_t = k)
        for all l = {1,...,L}
                t = {1,...,T}
                z_t = {1,...K}
                
        return an KXLXT matrix containing the gkl(t) 
        x_i is a Txd array having the observations in time
        """
        #get the time up to observations reach T
        T = x_i.shape[0]
        
        #get the number of states
        K = self.states_
        
        #get the number of gaussian components
        L = self.g_components_
        
        #initialize matrix
        gs = np.zeros( shape = [K, L, T] )
        
        for state in np.arange( K ):
            gs[state, :, :] = self.log_g_state( x_i, state )
            
        return gs
            
    
    def log_g_state(self, x_i, st = None):
        """
        computes the probability p(G_t = l|z_t = k, x_t)
        for state z_t = st and component G_t = comp
        for each time observation o_t t=1 ...T 
        for each component
        thus returning a :
        
        LxT matrix
        
        x_i = [x_i(1),...x_i(T)]
        x_i = ( Txd )
        """
        #number of gaussian components
        gauss_comp = self.g_components_
        
        #number of observations in time
        T = x_i.shape[0]
        
        #initialize the matrix to return
        log_pk = np.zeros( shape = [gauss_comp, T])
        
        for cmp in np.arange( gauss_comp ):
            log_pk[cmp, :] = self.log_predict_state_comp(x_i, st = st, cmp = cmp)
        
        #get the component wise sum for each sample in time
        log_sumPk = logsumexp( log_pk, axis = 0)
        
        #normalize the pk matrix such that every column sums to 0
        log_pk = log_pk-log_sumPk
        
        #checking if component wise they sum to 0 the gs
       #print("Check Gs")
        checkSum_zero(log_pk, axis = 0, name = "log_g_state")
        return log_pk
        
    #INITIALIZATIONS OF THE EM MODEL
    def EM_init(self, X, pi = None, A = None,alpha = None,
                                             means = None, cov = None):
        """
        Initialize the HMM parameters
        
        """
        idi = self.id
        
        print("Initializing the HMM with id:{}".format( idi ))
        
        self.pi_init( pi )
        self.A_init( A )
        self.alpha_means_cov_init(X, alpha, means, cov )
        self.gauss_init()
        self.initialized = True
        
        return self
        
    def pi_init(self, pi = None):
        """
        initialize initial state distribution randomly or
        with a custom matrix
        
        used by EM_init
        
        """
        
        if pi is not None:
            self.pi = pi
            
        else:
            
            pi = np.random.rand( self.pi.shape[0] )
            pi = pi/np.sum( pi )
            self.pi = pi
            
           # print("Check pi init")
            checkSum_one(self.pi, axis = 0, name = "pi_init")
            
        return self
        
        
    def A_init(self, A = None):
        """
        Initialize state transition matrix with a custom matrix A
        or randomly
        
        used by EM_init
        """
        
        if A is not None:
            self.A = A
        
        else:
            A = np.random.rand( self.A.shape[0], self.A.shape[1])
            Asum = np.sum ( A, axis = 1)
            A = A.T/Asum
            self.A = A.T
            
            #print("check A init")
            checkSum_one(self.A, axis = 1,name = "A_init")
        
        return self
            
    def alpha_means_cov_init(self, X, alpha, means, cov):
        """
        used by EM_init
        method
        Initializes alphas means and covs either with 
        custom matrix or with kmeans
        
        X = (N, T, d)
        
        """
       
        d = X.shape[2]
        K = self.states_
        L = self.g_components_
        
        #initialize matrices to zeros
        self.alpha = np.zeros( shape = [K, L])
        self.means = np.zeros( shape = [K, L, d])
        self.cov = np.zeros( shape = [K, L, d, d])
        
        init_type = self.gmm_init
        if (alpha is None) and (cov is None ) and ( means is None ):
            
            if init_type == "Kmeans":
                self.kmeans_init( X )
        
        if (alpha is not None) and (means is not None) and ( cov is not None):
            self.alpha = alpha
            self.means = means
            self.cov = cov
            
        return self
    
    def gauss_init(self):
        """
        Initializes the gaussian components
        
        """
        
        K = self.states_
        L = self.g_components_
        
        self.gauss = []
        for k in np.arange( K ):
            gaussState = []
            
            for l in np.arange( L ):
                mvg = multivariate_normal( mean = self.means[k,l,:], 
                                          cov = self.cov[k,l,:,:])
                
                gaussState.append(mvg)
            
            self.gauss.append( gaussState )
        
        return self
            
        
    def kmeans_init(self,  X ):
        """
        it is used from 
        "alpha_means_cov_init "
        method
        
        initiializes means with Kmeans
        covariances diagonal with variance 1
        alphas accordingly
        
        X = (N, T, d)
        
        """
        
        #take the number of points to use for Kmeans
        points = self.kmean_Points
        
        #number of points in dataset
        N = len( X ) 
        #number of samples in observation
        T = X[0].shape[0]
        #data dimension 
        d = X[0].shape[1] 
        #gaussian components
        L = self.g_components_
        #number of States
        K = self.states_
        
        
        #check points
        if points > N*T:
            points = N*T
            self.kmean_Points = points
        
        #if we initialize with -1 the kmean_Points take all points
        if points == -1:
            points = N*T
        #make the dataset to use in the kmeans
        X_make = self.make_dataset( X, points)
        
        #initialize Means 
        labels = self.kmeans_init_(X_make)
        
        #initialize alphas
        #number of points
        #N_x = len( labels )
        
        #find alphas
        #alphaK = np.zeros(K)
        #alphaL = np.zeros(L)
        for l in np.arange( K ):
            #indxl = len( np.where( labels == l )[0] )
            #indxl = len( np.where( labels == l )[0] )
            indx2 = np.where( labels == l )[0]
            Xl = np.var(X_make[indx2], axis = 0)
           # print(Xl.shape)
            self.cov[l, :] = np.diag(Xl)
            #alphaL[l] = indxl/N_x
            #alphaL[l] = 1/L
        
        #self.alpha[:] = alphaL
        self.alpha[:] = 1/L
        
        #checking sum of alphas on axis 1 KxL
        #print("check alphas init")
        checkSum_one(self.alpha, axis = 1, name = "init_alphas")
        
        #initialize Covarinaces
       # self.cov[:,:] = np.eye( d )
        
        return self
        
    
    def kmeans_init_(self, X_make):
        """ 
        this function is used by-->  "kmeans_init"
        Run the kmeans algorithms and sets the clsuter means
        X_make = [self.kmean_points, d]
        """
       
        #L = self.g_components_
        K = self.states_
        kmeans = KMeans( n_clusters = K )
        model = kmeans.fit(X_make)
        means = model.cluster_centers_
        labels = model.labels_
        
        for i in range(K):
            self.means[i, :] = means[i]
        
        return labels
        
        
    
    def make_dataset( self, X, points):
        """
        helper function for the Kmeans Initialization
        
        returns a dataset with points number of observations from X
        """
        T = X[0].shape[0]
        N = len( X )
        d = X[0].shape[1] 
        
        #see how many points we need to concatenate together
        indx_num = int( np.ceil( points/ T ) )
        #choose random indexes
        indx = np.random.choice( np.arange(N), size = indx_num, replace = False)
        indx = indx.astype(int)
        
        #return the Kmeans dataset
        X_kmeans = X[indx]
        X_kmeans = np.reshape( X_kmeans, [-1, d])
        
        return X_kmeans
        
           
    #EM UPDATE    
    def EM_iter(self, X, r_m, states = None,  pi = None, A = None,
                alpha = None, means = None, cov = None):
        """ 
        
        EM iteration updating all the 
        HMM parameteres
        
        A: state transition matrix
        pi: initial state probabilities
        alpha: gaussian mixing coefficients [k,l] matrix
        means: Gaussian Means [k,l,d] matrix
        cov: Coavriance matrices for the Gaussians [k,l,d,d]
        
        sets the A,pi, alpha, means, cov
        and gauss: gaussian components objects (frozen)
        
        X: is a list of obseravtions O^(1)... O^(N)
        O^(i) = [o_1^(i),....o_T^(i)]
        
        r_m: N dimensional matrix with the posterior probability p(H = m|X)
        which the probability of this HMM to be chosen given the observations
        X
        
        """
        
        #initialize EM if it is not initialized 
        if not self.initialized:
            print("Initialization in the EM_iter for HMM with id:{}".
                                                      format(self.id))
            
            self.EM_init(X, pi = pi, A = A, alpha = alpha, 
                                 means = means, cov = cov)
        
        K = self.states_
        L = self.g_components_
        
        #feature dimension
        d = X.shape[2]
       
        #initializing attributes for the EM
        self.initialize_EM_sums( K, L, d,  r_m )
        
        #r_m = np.log( r_m.copy())
        for i in np.arange( len(X) ):
            
            #print("iteration {} of internal EM for {} HMM".format(i, self.id))
            
            #get the ith observation
            x_i = X[i]
            
            #compute p_states, f start = time.time()
            log_p_states = self.log_predict_states_All(x_i)
            start = time.time()
            
            #puting states in forward
            if states is not None:
                si = states[i]
            else:
                si = None
                
            log_forw = self.log_forward(x_i, log_p_states = log_p_states, 
                                                                states = si )
            log_backw = self.log_backward(x_i, log_p_states = log_p_states)
            end = time.time() - start
            self.timeIn_p_States += end
            
            #get the gammas for the i_th observation KXT
            log_gamma_i = self.log_gamas(x_i, log_forw = log_forw, 
                                         log_backw = log_backw)
            
            #get xis for the i_th observation KXKX(T-1)
            log_xis_i = self.sliced(x_i, log_forw = log_forw,
                                    log_backw = log_backw, log_p_states = log_p_states)
            
            #get the rm_i,  N 
            rm_i = r_m[i]
            
            #get g_is for the ith observation KxLXT
            log_g_i = self.log_g(x_i)
            
            #update sum of pis
            self.update_pi( log_gamma_i[:, 0], rm_i )
            
            #update A matrix nominator and denominator
            self.update_A( log_xis_i, log_gamma_i, rm_i )
            
            #update alphas
            membs_i= self.update_alpha( np.exp(log_g_i), 
                                       np.exp(log_gamma_i), np.exp(rm_i))
            
            self.update_means_cov( x_i, membs_i)
        
        #set all the model parameteres
        self.set_EM_updates()
        print("Time in Pstates: {}".format( self.timeIn_p_States))
        
        return self
    
    
    #BEFORE EM ITERATION        
    def initialize_EM_sums( self, K, L, d,  log_r_m ):
        """
        initializes all the parameteres used in the EM_iter
        to be used in the inside for loop 
        if the dataset is way too big we might need to put 
        the observations in chuncks until we do a full EM update
        
        """
        
        #sum  of initial state distributions
        #for all the K states
        self.pi_Sum = np.full( shape = [K], fill_value = -np.math.inf)
        
        #sum of all rm_i's
        self.rm_Sum = logsumexp(log_r_m)
        
        #nominator for state transition probabilities
        self.A_nom = np.full( shape = [K,K], fill_value = -np.inf)
        
        #A denominator is the same as alpha denominator
        self.A_den = np.full( shape = [K], fill_value = -np.inf)
        
        #alpha_Nom is the same as means denominator
        #and covarinaces denominator
        #priors of the Gaussian components
        self.alpha_Nom = np.zeros( shape = [K, L] )
        self.alpha_Den = np.zeros( shape = [K])
        
        #nominator of the means
        self.means_Nom = np.zeros( shape = [K, L, d])
        #nominator of the covariances #need to subtract means
        self.cov_Nom = np.zeros( shape = [K, L, d, d] )
        
        return self
        
    
    #DURING THE FOR LOOP OF THE EM ITERATION        
    def update_pi(self,  gi1, rm_i):
        """ 
        updates the initial state parameter probabilities
        for all the states
        for the EM iteration
        given the values currently governed in the model
        p(z_1 = k|H = m)
        
        self.pi_Sum ---> the value to update
        
        USED IN EM_iter method
        """
        
        self.pi_Sum = logaddexp(gi1+rm_i, self.pi_Sum)
        
        #print("update_pi check gi1s")
        checkSum_zero(gi1, axis = 0, name = "update_pi : g1s")
        return
        
        
    def update_A(self, xis_i, gamma_i,  rm_i):
        """
        updates the sum of for the EM iteration
        given the values currently governed in the model
        self.A_nom
        self.A_den
        
        Aij = p(z_t = j| z_(t-1) = i)
        
        xis_i = KxKxT-1
        gamma_i = KxT
        """
        self.A_nom = logaddexp(logsumexp( xis_i, axis = 2)+rm_i, self.A_nom)
        #self.A_den += np.sum( gamma_i, axis = 1)*rm_i
        #self.A_den += np.sum(gamma_i[:, 0:-1], axis = 1)*rm_i
        #self.A_den =  xis_i.sum( axis = 1).sum(axis = 1)*rm_i
        #my = logaddexp(logsumexp( gamma_i[:,0:-1], axis = 1 )+rm_i, self.A_den)
        self.A_den = logaddexp(logsumexp(logsumexp( xis_i, axis = 1 ),axis = 1)+rm_i, self.A_den)

        return 
        
        
        
    def update_alpha(self, g_i, gamma_i, rm_i):
        """
        updates the mixing gaussian coefficients for each state
        p(G_t = l| z_t = k) = prior probability for the lth component 
        of k state
        
        this eventually will be a matrix KxL
        
        g_i KXLXT
        gamma_i KxT
        rm_i = scalar
        
        """
        #get the number of states K
        K  = g_i.shape[0]
        L = g_i.shape[1]
        T = g_i.shape[2]
        
        #holding the memberships for the T samples of i observation
        # of L compunents of K states
        memb_i = np.zeros( shape = [K, L, T])
        #for each state compute the priors for the lth components
        for k in np.arange( K ):
            memb_i[k, :, :] =  g_i[k, :, :]*gamma_i[k, :]*rm_i
            self.alpha_Nom[k, :] += memb_i[k,:,:].sum(axis = 1)
            
        
        self.alpha_Den += rm_i*gamma_i.sum( axis = 1 )
       
        
        return memb_i
    
    
    def update_means_cov(self, X_i,  membs_i):
        """
        Updates the nominator  of the mean 
        vectors of the Gaussians and partially updates the nominator
        of of the covarinace matrices
        
        means_Nom KXLXd
        cov_Nom KXLXdXd
        
        X_i ith observation TXd
        membs_i memberships of the samples of observation x_i KXLXT
        
        """
        K = membs_i.shape[0]
        L = membs_i.shape[1]
       
        
        for k in np.arange(K):
            for l in np.arange( L ):
                X_iw = membs_i[k, l, :]*(X_i.T) #dxT
                self.means_Nom[k, l, :] += np.sum( X_iw, axis = 1)
                self.cov_Nom[k, l, :, :] +=  (X_iw @ X_i) #dxd sum on all T
               
                
        return self
    
    #END OF THE EM ITERATION
    def set_EM_updates(self ) :
        """
        sets the new parameters of the HMM
        after one EM iteration
        
        pi K initial state distribution
        A KXK  state transition matrix
        alpha KXL gaussian priors
        means KxLxd means of the Gaussians
        cov KxLxdxd covarinaces of the Gaussians
        
        """
        #set pi
        logpi = self.pi_Sum-self.rm_Sum
        self.pi = np.exp(logpi)
       # print("check set pi")
        checkSum_one( self.pi, axis = 0, name = "set_pi")
        #set A
        logA = self.A_nom.T - self.A_den
        self.A = np.exp(logA.T)
       # print("check set A")
        checkSum_one( self.A, axis = 1, name = "set A")
        #set alpha
        self.alpha = ((self.alpha_Nom).T/self.alpha_Den).T
       # print("check set alpha")
        checkSum_one( self.alpha, axis = 1, name = "set alpha")
        #set means
        self.set_means()
        #set_covariances
        self.set_covs()
        #set gauss
        self.set_gauss()
        
    def set_means( self ):
        """
        set the means after the EM update
        meaning--> after finding means_Nom
        also prepares the covarinaces fortheir calculation in the next step
        of setEm_updates
        
        """
        
        K = self.alpha_Nom.shape[0]
        L = self.alpha_Nom.shape[1]
        
       
        for k in np.arange( K ):
            self.means[k, :, :] = ((self.means_Nom[k, :, :]).T \
                                                    /self.alpha_Nom[k,:]).T
            for l in np.arange( L ):
#                mN = self.means_Nom[k,l,:]
#                sN = self.cov_Nom[k,l,:]
#                mn2 = mN**2
#               w = self.alpha_Nom[k,l]
#                w2 = w**2
#                print( sN/w, mn2/w2)
                self.cov_Nom[k, l, :, :] = self.cov_Nom[k, l, :, :] \
                                                    /self.alpha_Nom[k,l]
                                                    
               
        return self
                      
    def set_covs( self ):
        """
        setting covariances 
        
        """
        reg = 10**(-3)
        K = self.alpha_Nom.shape[0]
        L = self.alpha_Nom.shape[1]
        
        for k in np.arange( K ):
            for l in np.arange( L ):
                meanKL = np.expand_dims( self.means[k,l,:], axis = 1).copy()
                self.cov[k, l, :, :] = self.cov_Nom[k,l,:,:] \
                - meanKL@(meanKL.T) + reg*np.eye( self.cov.shape[2])
#                print(self.id, self.cov[k,l,:,:], self.cov_Nom[k,l,:,:], meanKL)
                
        return self
    
    def set_gauss( self ):
        """
        after having compute means and covariances on the EM step
        setting the gaussian objects
        
        """
        
        K = self.means.shape[0]
        L = self.means.shape[1]
        
        gaussStates = []
        for k in np.arange(K):
            gaussComponents = []
            for l in np.arange(L):
                
                gaussComponents.append( 
                        multivariate_normal(mean = self.means[k,l,:],
                                            cov = self.cov[k,l,:,:]) )
            gaussStates.append( gaussComponents )
            
        self.gauss = gaussStates
        
        return self
    
    
    def get_params( self ):
        """
        Getting the parameteres of the HMM
        
        """
        
        alpha = self.alpha
        A = self.A
        cov = self.cov
        means = self.means
        pi = self.pi
        
        params = {'alpha':alpha, 'A': A, 'cov': cov, 'means': means,
                  'pi':pi}
        
        return params
                
    
         
         
         


class MHMM():
    """
    
    Mixture of HMMs class using the HMM class
    
    """
    
    def __init__(self, n_HMMS = 2, n_states = 2, n_Comp = 2, EM_iter = 10,
                 t_cov = 'diag', gmm_init = 'Kmeans', tol = 10**(-3)):
        
        self.gmm_init = 'Kmeans'
        #setting number of states per HMM
        self.states = n_states
        #setting the number of HMMS attribute
        self.n_HMMS = n_HMMS
        #setting the number of components of HMM attribute
        self.n_Comp = n_Comp
        #setting the covarinace type attribute
        self.t_cov = t_cov
        #settint the number of EMiterations attribute
        self.Em_iter = EM_iter
        #initializing n_HMMs for our mixture
        self.HMMS = []
        #mixing parameters of HMMs
        self.mix = np.zeros( n_HMMS)
        #initialize HMM classes
        self.HMM_init = False
        self.init_HMMs()
        #logLikelihood matrix for concergence
        self.logLikehood = np.zeros( self.Em_iter )
        self.initialized = False
        #tolerance in likelihood
        self._tol = tol
        
        
        
        
    def init_HMMs( self ):
        """ 
        
        Initializing  M HMMs from the HMM class
        into the 
        HMMS list attribute 
        
        """
        
        M = self.n_HMMS
        states = self.states
        n_Comp = self.n_Comp
        t_cov = self.t_cov
        gmm_init = self.gmm_init
        
        for m in np.arange( M ):
            self.HMMS.append( HMM( states = states, g_components = n_Comp,
                                  t_cov = t_cov, gmm_init = gmm_init, idi = m))
            
        self.HMM_init = True
            
        return self
    
    def EM_init(self, X, mix = None, pi = None, A = None,  alpha = None,
                means = None, cov = None):
        
        """
        Initialize the model for the EM algorithm
        
        X = [N,T,d] array of observations
        mix = (M,) probability of using mth HMM
        pi = [M,K] initial state probabilities of the K states of
             the M HMMs
        A = [M,K,K] matrix with M state transition matrices of size KxK
        alpha = [M,K,L] mixture components of the L gaussians of the K states
                        of the M hmms
        
        means = [M,K,L,d] [K,L,d] gaussians means matrices for the M HMMs
        cov = [M,K,L,d,d] gaussian covariances matrices for the M HMMs
        
        """
        M = self.n_HMMS
        
        if mix is None:
            mix = np.random.rand(M)
            mSum = np.sum(mix)
            self.mix = mix/mSum
            
        for m in np.arange( M ):
            
            self.HMMS[m].EM_init( X, pi = pi, A = A, alpha = alpha,
                                             means = means, cov =cov)
            
        #set the attribute that the modle was initialized    
        self.initialized = True
        
        return self
            
    
    #BEGIN OF EM FIT
    def fit( self, data = None, mix = None, pi = None, A = None,  alpha = None,
                means = None, cov = None, states = None):
        
        """
        
        Fit the Mixture of HMMs with the EM algorithm
        
        sup_states_ = NxT matrix containing the known states of each time series
        """
        
        
        if data is None:
            print("Error no data to fit")
            return
        
        #number of HMM iterations
        em_iter = self.Em_iter
        
        #initialize EM algorithm
        if not self.initialized:
            
            self.EM_init(data, mix = None, pi = None, A = None,  alpha = None,
                means = None, cov = None)
        
        for iter1 in  np.arange( em_iter ):
            print("Iteration {} of EM".format( iter1 ))
            self.EM_update( data, states = states  )
            
            if self.convergenceMonitor(data, iter1) :
                break
            
        return self
        
        
    
    def EM_update(self, X, states = None):
        """
        performs the EM update for the mixture of HMMs
        
        used by fit function
        
        """
        
        #take the number of HMMs
        M = self.n_HMMS
        R = self.log_posterior_All( X, states = states )
        #print("Checking R")
        #update the mixing parameters
        self.update_mix(R)
        
        for m in np.arange( M ):
            print("Training the {}th HMM".format(m))
            hmm_m = self.HMMS[m]
            hmm_m.EM_iter(X, R[:, m], states = states )
            
        return self
        
        
        
    def update_mix(self, R):
        """
        updates the mixing parameteres
        of the HMMs
        R = (N,M)
        """
        N = R.shape[0]
        self.mix = np.sum( np.exp(R) , axis = 0)/N
        
        return self
    #END OF EM FIT
        
    def predict_log_proba(self, x_i ):
        """
        predicts the probability of an observation given the model
        
        p(x_i ; M) = Sum_{i = 1...M} p(x_i, H = m;M)
                   = Sum_{i=1...M} p(x_i|H=m;M)p(H = m)
                   
        x_i = (T,d)
                   
        """
        
        M = self.n_HMMS
        mix = np.log(self.mix)
        px_i = -np.math.inf
        
        for  m in np.arange( M ):
            hmm_m = self.HMMS[m]
            px_i = logaddexp(hmm_m.log_predict_x( x_i )+mix[m], px_i)
            
        return px_i
    
    
    def log_posterior_HMM(self, x_i, states = None):
        """
        calculates the posterior of each HMM 
        p(H = m | x_i;Model) 
        for every m
        
        rx_i = (M,)
        """
        
        M = self.n_HMMS
        mix = np.log(self.mix)
        rx_i = np.zeros(M)
        
        for m in np.arange( M ):
            hmm_m = self.HMMS[m]
            rx_i[m] = hmm_m.log_predict_x( x_i, states = states )+mix[m]
            
        sum_rx_i = logsumexp( rx_i )
        #normalize posteriors
        rx_i = rx_i-sum_rx_i
        
        #print("Checking MHMM posterior")
        checkSum_zero(rx_i, axis = 0, name = "MHMM posterior")
        
        return rx_i
            
        
    def log_posterior_All(self, X, states = None):
        """
        
        predicts the posterior probabilituy of being in an HMM
        for all X_i in X
        
        X = [N,T,d] matrix
        
        returns Matrix with the posteriors
        R = [N,M] for each HMM
        
        """
        
        N = X.shape[0]
        M = self.n_HMMS
        R = np.zeros( shape = [N,M] )
        
        
        
        for i in np.arange(N):
            if states is None:
                si = None
            else:
                si = states[i]
                
            R[i,:] = self.log_posterior_HMM(X[i], states = si)
        
        return R
    
    def get_params( self ):
        """
        Gets the Parameteres of The individual HMMs
        
        """
        M = self.n_HMMS
        
        params = []
        mix = self.mix
        
        for m in np.arange( M ):
            params.append(self.HMMS[m].get_params())
            
        params_All = {'mix':mix, 'params':params}
        
        return params_All
    
    def convergenceMonitor(self, data, iteration):
        """
        Computes the Log Likelihood of the Data under the model
        and updates the Loglikelihood matrix
        
        """
        break_condition = False
        N = data.shape[0]
        i = iteration
        tol = self._tol
        for n in np.arange( N ):
            self.logLikehood[i] +=  self.predict_log_proba( data[n] )
        
        self.logLikehood[i]  = self.logLikehood[i]/N
        lgi = self.logLikehood[i]
        
        print("Iteration: {} LogLikelihood:{:.2}".format( i, lgi ))
        
        if i > 0:
            diff = np.abs( self.logLikehood[i] - self.logLikehood[i-1])
        else:
            diff = 1000
            
        if diff < tol:
            break_condition = True
            print("Convergence Criteria has been met")
            
        return break_condition
        
        
   
    
    
    
    
    
    
    





