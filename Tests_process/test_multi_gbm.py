# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:09:37 2026

@author: Nicola
"""

import numpy as np 
import pytest 

from process.multivariate_process import MultiGBM 


class TestsMultiGBM:
    
    def test_multiGBM(self):
        
        rho = -.1
        sigma1 = .2 
        sigma2 = .3 
        
        SIGMA = np.array([[sigma1**2,         sigma1*sigma2*rho], 
                          [sigma1*sigma2*rho, sigma2**2]])
        
        MU = np.array([0, 0])
        
        X0 = np.array([1, 1])
        
        T = 1
        n_steps = 10
        
        N = 13
        
        model = MultiGBM(MU, SIGMA)
        
        trj = model.simulate(X0, T, n_steps, N)
        
        martingale_check= np.mean(trj, axis=0) 
        
        for i in range(0, n_steps+1):
            
            np.testing.assert_allclose(martingale_check[i], X0, atol=1e-2, rtol=1e-2)