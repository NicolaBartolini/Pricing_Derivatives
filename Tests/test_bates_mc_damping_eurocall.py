# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:52:11 2026

@author: Nicola
"""

import sys 

sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance") 

import numpy as np
from datetime import datetime,timedelta


from process_class import Bates
from EuropeanOption_class import EuroCall
from FourierDampingEngine import FourierDampingEngine
from MonteCarloEngine import MonteCarloEngine


# (mu, kappa, theta, eta, rho, lambda, muJ, sigmaJ)
PARAMETERS = [

    (0.0, 2.0, 0.04, 0.30, -0.70, 0.20, -0.05, 0.20),
    (0.0, 1.5, 0.05, 0.45, -0.50, 0.40, -0.08, 0.30),
    (0.0, 3.0, 0.09, 0.25, -0.30, 0.60, 0.02, 0.25)
]
 


def test_bates_mc_vs_Damping():
    
    print("\n")
    
    S0 = 1.0
    v0 = .05
    
    r = 0.0
    
    n_steps = 252
    n_paths = 18
    
    # 1 year maturity option
    today = datetime(2026,1,1)
    maturity = today + timedelta(days=int(365))
    
    for mu, kappa, theta, eta, rho, lam, muj, sigmaj in PARAMETERS:
        
        model = Bates(r, kappa, theta, eta, rho, lam, muj, sigmaj)
        
        for strike in [.8, 1, 1.2]: 
            
            option = EuroCall(strike, maturity) 
            
            mc_price = MonteCarloEngine(option, model, [S0,v0], r, n_steps=n_steps, n=n_paths, day=today)
            
            fourier_price = FourierDampingEngine(option, model, [S0,v0], r, alpha=1.5, day=today)

            print()
            print("MC =",mc_price)
            print("Fourier =",fourier_price)
            print("Diff =",mc_price-fourier_price)


            np.testing.assert_allclose(mc_price, fourier_price, rtol=5e-2, atol=5e-2)
            
            
            

