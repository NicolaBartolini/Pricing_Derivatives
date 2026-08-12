# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 08:08:58 2026

@author: Nicola
"""

import os
import sys

root = os.getcwd().split("\\")[:-1]

sys.path.append(os.path.join("\\".join(root)))

from process_class import Merton 
import numpy as np
import pytest




# (S0, mu, sigma, lambda, muJ, sigmaJ)
PARAMETERS = [

    (1,    0.0, .2,  0.20, -0.05, 0.20),
    (3.75, 0.0, .5,  0.40, -0.08, 0.30),
    (5.7,   0.0, .35, 0.60,  0.02, 0.25)]

np.random.seed(42)

def test_bates_martingale():
    
    # S0 = 1.0
        
    n_steps = 252
    n_paths = 19
    
    T = 1
    
    for S0, mu, sigma, lam, muj, sigmaj in PARAMETERS:
        
        model = Merton(mu, sigma, lam, muj, sigmaj)
        
        paths = model.simulate(X0=[S0], T=T, n_steps=n_steps, n=n_paths)
        
        # Monte Carlo estimate of E[S_T]
        sample_mean = np.mean(paths[-1]) 
        
        print(f"\nS0 : {S0} \t S  : {sample_mean}")
        
        # Check martingale property
        assert abs(sample_mean - S0) < 1e-2

