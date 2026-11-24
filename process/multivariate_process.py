# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:08:28 2026

@author: Nicola
"""

import numpy as np 


class MultiGBM:
    # Multivariate GBM model 
    
    def __init__(self, MU, SIGMA):
        
        # MU    = np.array with shape (1, n) or (n,) - drift
        # SIGMA = np.array with shape (n, n) - matrix whose entries are the vol of each single BM (no correlation here) the simplest way is SIGMA be diagonal
        # CORR = np.array with shape (n,n) - matrix of correlations so the diagonal has only 1
        
        self.MU = MU 
        self.SIGMA = SIGMA 
        self.n_dim = len(SIGMA)
        # self.CORR = CORR
    
    def simulate(self, S0 : np.array, T : float, n_steps : int, N :int) -> np.array:
        
        n = 1<<N
        Xt = np.empty((n, n_steps + 1, self.n_dim))
        # St[:, 0, :] = X0
        Xt[:,0,:] = 0 
        dt = T/n_steps
        sqrt_dt = np.sqrt(dt)
        chol_SIGMA = np.linalg.cholesky(self.SIGMA)
        
        compensator = np.sum(self.SIGMA, axis=1)
        
        drift = (self.MU - .5 * compensator) * dt
        
        # compensator = np.sum(chol_SIGMA, axis=1)
        # compensator = np.reshape(compensator, (1, len(compensator)))
        
        for i in range(0, n, 2):
        
            # I make steps of 2 in order to use antithetic variate
            
            # BM1 = sqrt_dt * chol_SIGMA @ np.random.normal(0, 1, (n_steps, self.n_dim)) # for step i
            
            for t in range(1, n_steps+1):
                
                BM1 = sqrt_dt * chol_SIGMA @ np.random.normal(0, 1, (self.n_dim, 1)) # for step i
                
                Xt[i, t, :] = Xt[i, t-1, :] + drift + BM1.flatten()
                Xt[i+1, t, :] = Xt[i+1, t-1, :] + drift - BM1.flatten() # antithetic variate (-BM1 is also a BM since the Gaussian distribution is symmetric)
                
               
        return S0 * np.exp(Xt) 
    

if __name__=='__main__':
    
    np.random.seed(5)
    
    rho = -.1
    sigma1 = .2 
    sigma2 = .3 
    
    SIGMA = np.array([[sigma1**2,         sigma1*sigma2*rho], 
                      [sigma1*sigma2*rho, sigma2**2]])
    
    MU = np.array([[0, 0]])
    
    X0 = np.array([[1, 1]])
    
    T = 5
    n_steps = 252 
    
    N = 10
    
    model = MultiGBM(MU, SIGMA)
    
    trj = model.simulate(X0, T, n_steps, N)
    
    martingale_check= np.mean(trj, axis=0)
    
    #### char fun check 
    
    u = np.array([[1, 1]])
    