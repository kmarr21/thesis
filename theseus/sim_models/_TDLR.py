import numpy as np
from theseus.helper import inv_logit


class TDLR(object):
    
    def __init__(self, beta_1, beta_2, alpha_com, alpha_rare):
        
        ## Define parameters.
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha_com = alpha_com
        self.alpha_rare = alpha_rare
        
        ## Initialize Q-values.
        self.Q = None
        
    def train(self, R, T=[[0.7,0.3],[0.3,0.7]], reset=False):
        
        ## Error-catching: rewards.
        R = np.array(R)
        
        ## Error-catching: transition probabilities.
        T = np.array(T)
        
        ## Initialize Q-values.
        if self.Q is None or reset:
            self.Q = 0.5 * np.ones((3,2))
            
        ## Preallocate space.
        n_trials = R.shape[0]
        Y = np.zeros((n_trials, 2), dtype=int)
        t = np.zeros(n_trials, dtype=int)
        r = np.zeros(n_trials)
            
        for i in range(n_trials):
            
            ## Stage 1: Re-compute Q-values.
            #self.Q[0] = T @ self.Q[1:].max(axis=1)
            self.Q[0,0] = 0.7*max(self.Q[1]) + 0.3*max(self.Q[2])
            self.Q[0,1] = 0.3*max(self.Q[1]) + 0.7*max(self.Q[2])
            
            ## Stage 1: Compute choice likelihood.
            theta = inv_logit( self.beta_1 * np.diff(self.Q[0]) )
            
            ## Stage 1: Simulate choice.
            Y[i,0] = np.random.binomial(1,theta)
            
            ## Simulate transition.
            t[i] = np.random.binomial(1, 0.7)
            S = np.where(t[i], Y[i,0], 1-Y[i,0]) + 1
                        
            ## Stage 2: Compute choice likelihood.
            theta = inv_logit( self.beta_2 * np.diff(self.Q[S]) )
            
            ## Stage 2: Simulate choice.
            Y[i,1] = np.random.binomial(1,theta)
            
            ## Stage 2: Observe outcome.
            r[i] = R[i,S-1,Y[i,1]]
            
            # Check for transition type and assign LR
            if Y[i,0]+1 == S:
                alpha = self.alpha_com
            else:
                alpha = self.alpha_rare
            
            ## Stage 2: Update Q-values.
            self.Q[S,Y[i,1]] += alpha * ( r[i] - self.Q[S,Y[i,1]] )
            
        return Y, t, r

        
