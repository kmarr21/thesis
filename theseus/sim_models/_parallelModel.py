import numpy as np
from theseus.helper import inv_logit


# Hybrid model written as described by Toyama
class ParallelModel(object):
    
    def __init__(self, b1, b2, a1, a2, w, lam, p):
        
        ## Define parameters.
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.w = w
        self.l = lam
        self.p = p

        ## Initialize Q-values.
        self.MB = None
        self.MF = None
        self.HYB = None
        
    def train(self, R, T=[[0.7,0.3],[0.3,0.7]], reset=False):
        
        ## Error-catching: rewards.
        R = np.array(R)
        
        ## Error-catching: transition probabilities.
        T = np.array(T)
        
        ## Initialize Q-values.
        if self.MB is None or reset:
            self.MB = 0.5 * np.ones(2)
        
        if self.MF is None or reset:
            self.MF = 0.5 * np.ones((3,2))
        
        if self.HYB is None or reset:
            self.HYB = 0.5 * np.ones(2)
            
        ## Preallocate space.
        n_trials = R.shape[0]
        Y = np.zeros((n_trials, 2), dtype=int)
        t = np.zeros(n_trials, dtype=int)
        r = np.zeros(n_trials)
            
        for i in range(n_trials):
             
            # compute first stage model-based values based on transition
            self.MB[0] = 0.7*max(self.MF[1]) + 0.3*max(self.MF[2])
            self.MB[1] = 0.3*max(self.MF[1]) + 0.7*max(self.MF[2])
            
            self.HYB[0] = self.w * self.MB[0] + (1 - self.w) * self.MF[0,0]
            self.HYB[1] = self.w * self.MB[1] + (1 - self.w) * self.MF[0,1]
            
            hybrid1 = self.HYB[1] - self.HYB[0]
            
            ## Stage 1: Compute choice likelihood.
            if i==0:
                theta = inv_logit( self.b1 * hybrid1 )
            else:
                m = -1 if Y[i-1,0] == 0 else 1
                theta = inv_logit( self.b1 * hybrid1 + self.p*m )

            ## 1 (0) => pi = -1
            ## 2 (1) => pi = 1
            
            ## Stage 1: Simulate choice.
            Y[i,0] = np.random.binomial(1,theta)
            
            ## Simulate transition.
            t[i] = np.random.binomial(1, 0.7)
            S = np.where(t[i], Y[i,0], 1-Y[i,0]) + 1
            
            hybrid2 = self.MF[S,1] - self.MF[S,0]
                        
            ## Stage 2: Compute choice likelihood.
            theta = inv_logit( self.b2 * hybrid2 )
            
            ## Stage 2: Simulate choice.
            Y[i,1] = np.random.binomial(1,theta)
            
            ## Stage 2: Observe outcome.
            r[i] = R[i,S-1,Y[i,1]]
            
            # first and second stage Q-MF updates
            self.MF[0, Y[i,0]] += self.a1 * ( self.MF[S,Y[i,1]] - self.MF[0, Y[i,0]] )
            
            self.MF[S,Y[i,1]] += self.a2 * ( r[i] - self.MF[S, Y[i,1]] )
            
            self.MF[0, Y[i,0]] += self.l * self.a1 * ( r[i] - self.MF[S, Y[i,1]] )
                                                                                                                                               
            
        return Y, t, r
