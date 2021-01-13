import numpy as np
from theseus._helper import inv_logit


class Hybrid(object):
    
    def __init__(self, b1, b2, a1, a2, w, p):
        
        ## Define parameters.
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.w = w
        self.p = p
        
        ## Initialize Q-values.
        self.MB = None
        self.MF = None
        
    def train(self, R, reset=False):
        
        ## Error-catching: rewards.
        R = np.array(R)
        
        ## Initialize Q-values.
        if self.MB is None or reset:
            self.MB = 0.5 * np.ones((2,2))
        
        if self.MF is None or reset:
            self.MF = 0.5 * np.ones(2)
            
        
        ## Preallocate space.
        n_trials = R.shape[0]
        Y = np.zeros((n_trials, 2), dtype=int)
        t = np.zeros(n_trials, dtype=int)
        r = np.zeros(n_trials)
            
        for i in range(n_trials):
             
            hybrid1 = self.w * 0.4 * (max(self.MB[1]) - max(self.MB[0])) + (1-self.w)*(self.MF[1] - self.MF[0])
            
            ## Stage 1: Compute choice likelihood.
            if i == 0:
                theta = inv_logit( self.b1 * hybrid1 )
            else:
                m = -1 if (Y[i-1,0] == 0) else 1
                theta = inv_logit( self.b1 * hybrid1 + self.p*m )

            ## 1 (0) => pi = -1
            ## 2 (1) => pi = 1
            
            ## Stage 1: Simulate choice.
            Y[i,0] = np.random.binomial(1,theta)
            
            ## Simulate transition.
            t[i] = np.random.binomial(1, 0.7)
            S = np.where(t[i], Y[i,0], 1-Y[i,0]) + 1
            
            hybrid2 = self.MB[S-1,1] - self.MB[S-1,0]
                        
            ## Stage 2: Compute choice likelihood.
            theta = inv_logit( self.b2 * hybrid2 )
            
            ## Stage 2: Simulate choice.
            Y[i,1] = np.random.binomial(1,theta)
            
            ## Stage 2: Observe outcome.
            r[i] = R[i,S-1,Y[i,1]]
            
            ## Update Model-Free values
            self.MF[Y[i,0]] += self.a1 * (self.MB[S-1,Y[i,1]] - self.MF[Y[i,0]]) + self.a1*(r[i] - self.MB[S-1,Y[i,1]])
            
            ## Update Model-Based values
            self.MB[S-1,Y[i,1]] += self.a2 * (r[i] - self.MB[S-1, Y[i,1]])
            
        return Y, t, r
        
