# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:24:08 2026

@author: Nicola
"""

import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))

from process_class import Bates 
import numpy as np
import pytest




# (mu, kappa, theta, eta, rho, lambda, muJ, sigmaJ)
PARAMETERS = [

    (0.0, 2.0, 0.04, 0.30, -0.70, 0.20, -0.05, 0.20),
    (0.0, 1.5, 0.05, 0.45, -0.50, 0.40, -0.08, 0.30),
    (0.0, 3.0, 0.09, 0.25, -0.30, 0.60, 0.02, 0.25)
]


def test_bates_martingale():
    
    S0 = 1.0
    nu0 = .05
    
    # mu = 0.0
    
    n_steps = 252
    n_paths = 18
    
    T = 1
    
    for mu, kappa, theta, eta, rho, lam, muj, sigmaj in PARAMETERS:
        
        model = Bates(mu, kappa, theta, eta, rho, lam, muj, sigmaj)
        
        paths = model.simulate(X0=[S0, nu0], T=T, n_steps=n_steps, n=n_paths)
        
        # Monte Carlo estimate of E[S_T]
        sample_mean = np.mean(paths[0][-1]) 
        
        print(f"\nS0 : {S0} \t S  : {sample_mean}")
        
        # Check martingale property
        assert abs(sample_mean - S0) < 1e-1
