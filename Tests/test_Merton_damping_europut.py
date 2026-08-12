# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 08:36:00 2026

@author: Nicola
"""


import os
import sys

root = os.getcwd().split("\\")[:-1]

sys.path.append(os.path.join("\\".join(root)))

import numpy as np
from datetime import datetime,timedelta


from process_class import Merton
from EuropeanOption_class import EuroPut
from FourierDampingEngine import FourierDampingEngine
from MonteCarloEngine import MonteCarloEngine


# (mu, sigma, lambda, muJ, sigmaJ)
PARAMETERS = [

    (0.02, .2, 0.20, -0.05, 0.20),
    (0.02, .3, 0.40, -0.08, 0.30),
    (0.02, .5, 0.60, 0.02, 0.25)
]
 


def test_bates_mc_vs_Damping():
    
    print("\n")
    
    S0 = 1.0
    v0 = .05
    
    r = 0.02
    
    n_steps = 252
    n_paths = 18
    
    # 1 year maturity option
    today = datetime(2026,1,1)
    maturity = today + timedelta(days=int(365))
    
    for mu, sigma, lam, muj, sigmaj in PARAMETERS:
        
        model = Merton(r, sigma, lam, muj, sigmaj)
        
        for strike in [.8, 1, 1.2]: 
            
            option = EuroPut(strike, maturity) 
            
            mc_price = MonteCarloEngine(option, model, [S0,v0], r, n_steps=n_steps, n=n_paths, day=today)
            
            fourier_price = FourierDampingEngine(option, model, [S0,v0], r, alpha=-1.5, day=today)

            print()
            print("MC =",mc_price)
            print("Fourier =",fourier_price)
            print("Diff =",mc_price-fourier_price)


            np.testing.assert_allclose(mc_price, fourier_price, rtol=5e-2, atol=5e-2)
            
            
            

