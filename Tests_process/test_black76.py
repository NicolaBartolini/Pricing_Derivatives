# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:05:36 2026

@author: Nicola
"""

import numpy as np 
import pytest 

from process.process_class import Black76 

np.random.seed(2) 


class TestBlack76:
    
    @pytest.mark.parametrize('T', np.arange(1, 6, 1))
    @pytest.mark.parametrize('sigma', np.arange(.2, 1, .2))
    def test_martingale(self, T, sigma):
        
        L0 = 1 # initial value
        N = 18 # 2^N monte-carlo simulations 
        n_steps = 250 # number of steps
        
        # initialize the Black76 model 
        model = Black76(sigma)
        
        # get the trajectory at maturity T 
        Lt = model.simulate([L0], T, n_steps, N)[-1]
        
        # compute the mean
        sample_mean = np.mean(Lt)
        
        np.testing.assert_allclose(sample_mean, L0, rtol=1e-2, atol=1e-2)
        
        