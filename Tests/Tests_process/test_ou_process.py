# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:06:38 2026

@author: Nicola
"""

import numpy as np
import pytest 

from process.additive_process import OrnsteinUhlenbeck 


np.random.seed(42)



def vasicek_conditional_mean(x, t, T, kappa, theta):
    """E[X_T | X_t = x] for the Vasicek model."""
    tau = T - t
    return theta + (x - theta) * np.exp(-kappa * tau)


def vasicek_conditional_variance(t, T, kappa, sigma):
    """Var[X_T | X_t = x] for the Vasicek model."""
    tau = T - t
    return sigma**2 / (2.0 * kappa) * (1 - np.exp(-2.0 * kappa * tau))

 

class TestOUprocess:
    
    @pytest.fixture
    def model_instance(self):
        """Fixture providing shared baseline market and option parameters."""
        # params =  {'x0' : .01, 'kappa' : 1.5, 'theta' : .025, 'sigma' : .2}
        
        model = OrnsteinUhlenbeck(kappa=1.5, theta=.025, sigma=.2)
    
        return model 
    
    @pytest.mark.parametrize('T', range(1, 11))
    def test_mean(self, model_instance, T):
        
        x0 = .01
        n_steps = T * 200 
        n = 12
        
        Xt = model_instance.simulate([x0], T, n_steps*T, n)[-1]
        
        sample_mean = np.mean(Xt)
        
        exact_mean = vasicek_conditional_mean(x0, 0, T, model_instance.kappa, model_instance.theta)
        
        np.testing.assert_allclose(sample_mean, exact_mean, rtol=1e-2, atol=1e-2)
    
    
    @pytest.mark.parametrize('T', range(1, 11))
    def test_variance(self, model_instance, T):
        
        x0 = .01
        n_steps = T * 200 
        n = 12
        
        Xt = model_instance.simulate([x0], T, n_steps*T, n)[-1]
        
        sample_variance = np.var(Xt)
        
        exact_variance = vasicek_conditional_variance(0, T, model_instance.kappa, model_instance.sigma)
        
        # print(exact_variance)
        # print(sample_variance)
        
        np.testing.assert_allclose(sample_variance, exact_variance, rtol=1e-2, atol=1e-2)
        
        
        
        
        