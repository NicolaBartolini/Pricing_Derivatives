# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:19:42 2026

@author: Nicola
"""

import numpy as np 
import pytest 

from process.process_class import Sabr 

np.random.seed(42)
# np.random.seed(1)

class TestSabr:
    
    @pytest.mark.parametrize('T', [1])
    @pytest.mark.parametrize('sigma', [.25, .5])
    @pytest.mark.parametrize('beta', [.2, .5])
    @pytest.mark.parametrize('rho', [-.1, .1])
    def test_martingale_sabr(self, T, sigma, beta, rho):
        
        X0 = [1, .2] # initial value
        N = 13 # 2^N monte-carlo simulations 
        n_steps = 252 * T # number of steps
        
        # initialize the Black76 model 
        model = Sabr(beta, sigma, rho)
        
        # get the trajectory at maturity T 
        Lt = model.simulate(X0, T, n_steps, N)[0][-1]
        
        Xt = model.simulate(X0, T, n_steps, N)
        
        Lt = Xt[0]
        nut = Xt[1]
        
        # compute the mean
        sample_mean_Lt = np.mean(Lt[-1])
        sample_mean_nut = np.mean(nut[-1])
        
        np.testing.assert_allclose(sample_mean_Lt, X0[0], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(sample_mean_nut, X0[1], rtol=1e-2, atol=1e-2)