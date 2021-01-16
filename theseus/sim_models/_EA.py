import numpy as np
from theseus._helper import inv_logit

class EA(object):
    
    def __init__(self, b1, b2, eta1, eta2, w, p):
        
        ## Define parameters.
        self.b1 = b1
        self.b2 = b2
        self.eta1 = eta1
        self.eta2 = eta2
        self.w = w
        self.p = p
        
        ## Initialize Q-values.
        self.Qs1 = None
        self.Qs2 = None
        
    def train(self, R, T=[[0.7,0.3],[0.3,0.7]], reset=False):
        
        ## Error-catching: rewards.
        R = np.array(R)
        
        ## Error-catching: transitions.
        T = np.array(T)
        
        ## Initialize Q-values.
        if self.Qs1 is None or reset:
            self.Qs1 = 0.5 * np.ones(2)
        
        if self.Qs2 is None or reset:
            self.Qs2 = 0.5 * np.ones((2,2))
            
        
        ## Preallocate space.
        n_trials = R.shape[0]
        Y = np.zeros((n_trials, 2), dtype=int)
        t = np.zeros(n_trials, dtype=int)
        r = np.zeros(n_trials, dtype=int)
        X = np.zeros(n_trials, dtype=int)
            
        for i in range(n_trials):

            theta1 = inv_logit( self.b1 * (self.Qs1[1] - self.Qs1[0]))
            
            ## Stage 1: Simulate choice.
            Y[i,0] = np.random.binomial(1,theta1)
            
            ## Simulate transition.
            t[i] = np.random.binomial(1, 0.7)
            S = np.where(t[i], Y[i,0], 1-Y[i,0]) + 1
            X[i] = S-1
                        
            ## Stage 2: Compute choice likelihood.
            theta2 = inv_logit( self.b2 * (self.Qs2[X[i],1] - self.Qs2[X[i],0]) )
            
            ## Stage 2: Simulate choice.
            Y[i,1] = np.random.binomial(1,theta2)
            
            ## Stage 2: Observe outcome.
            r[i] = R[i,S-1,Y[i,1]]
            
            ## Update stage 2 Q-values
            self.Qs2[X[i], Y[i,1]] += self.eta2 * (r[i] - self.Qs2[X[i], Y[i,1]])
            # where's eta1 in all this?
            
            ## Update stage 1 Q-values
            if (Y[i,1] == 0):
                # if s1 option 1 chosen
                # Chosen update:
                self.Qs1[0] = self.w * (self.Qs1[0] + T[X[i], 0] * (r[i] - self.Qs1[0])) + (1 - self.w)*(r[i] - self.Qs1[0])
                # Unchosen update:
                self.Qs1[1] = self.w * (self.Qs1[1] + T[X[i], 1] * (r[i] - self.Qs1[1])) + (1 - self.w)*(self.Qs1[1])
            else:
                # if s1 option 2 chosen
                # Chosen update:
                self.Qs1[1] = self.w * (self.Qs1[1] + T[X[i], 1] * (r[i] - self.Qs1[1])) + (1 - self.w)*(r[i] - self.Qs1[1])
                # Unchosen update:
                self.Qs1[0] = self.w * (self.Qs1[0] + T[X[i], 0] * (r[i] - self.Qs1[0])) + (1 - self.w)*(self.Qs1[0])
                
        return Y, t, r, X
