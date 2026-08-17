# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:56:30 2026

@author: Nicola
"""
import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))

from process_class import GBM 
import numpy as np
import pytest


@pytest.mark.parametrize("sigma", np.linspace(0.1, 1, 10))

@pytest.mark.parametrize("T", np.arange(.25, 1.25, .25) ) 


def test_gbm_martingale(sigma, T):

    S0 = 100.0
    mu = 0.0

    n_steps = 100
    n_paths = 14

    gbm = GBM( mu=mu, sigma=sigma)

    paths = gbm.simulate(X0=[S0], T=T, n_steps=n_steps, n=n_paths)

    # Monte Carlo estimate of E[S_T]
    sample_mean = np.mean(paths[-1]) 
    
    # Theoretical expectation
    theoretical_mean = S0 * np.exp(mu * T)

    # Theoretical variance
    theoretical_variance = (S0**2* np.exp(2 * mu * T)* (np.exp(sigma**2 * T) - 1))

    # Monte Carlo standard error
    standard_error = np.sqrt(theoretical_variance / n_paths)

    # Check martingale property
    assert abs(sample_mean - theoretical_mean) < 4 * standard_error
