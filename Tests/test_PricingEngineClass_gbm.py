# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:02:28 2026

@author: Nicola
"""

import os  
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append("\\".join(root))

from EuropeanOption_class import EuroCall 
from process_class import GBM 
from PricingEngine_class import * 

import numpy as np 
from datetime import datetime, timedelta 
from scipy.stats import norm

import pytest 


def black_scholes_call(S0, K, r, sigma, T):
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

np.random.seed(42)

@pytest.mark.parametrize("sigma", np.linspace(0.10, 0.50, 2))
@pytest.mark.parametrize("T", [.5, 1.00])
def test_fourier_engine_matches_black_scholes(sigma, T):

    S0 = 100.0
    K = 100.0
    r = 0.05
    alpha = 1.5
    n = 20
    n_steps = int(365 * T)
    X0 = [S0]

    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=n_steps)

    option = EuroCall(K, maturity)
    process = GBM(r, sigma) 
    
    MC_pricer = MonteCarloPricer(n_steps, n)
    CarrMadan_pricer = FourierDampingPricer(alpha)
    COS_pricer = CosPricer()
    
    
    exact_price = black_scholes_call(S0, K, r, sigma, T)
    # mc_price = MC_pricer.evaluate_option(option, process, X0, r, today)
    # damping_price = CarrMadan_pricer.evaluate_option(option, process, X0, r, today) 
    # cos_price = COS_pricer.evaluate_option(option, process, X0, r, today)
    
    # Dictionary of engines and their acceptable absolute tolerances
    # Note: Monte Carlo has higher variance than analytical COS / Fourier methods
    pricers = {"COS": (COS_pricer, 1e-1),
               "FourierDamping": (CarrMadan_pricer, 1e-1),
               "MonteCarlo": (MC_pricer, 1e-1)}
    
    print()
    
    for name, (pricer, tol) in pricers.items():
        computed_price = pricer.evaluate_option(option, process, X0, r, today)
        
        print(f"Exact        : {exact_price}")
        print(f"Engine price : {computed_price}")
        print(f"Diff         : {abs(exact_price - computed_price)}")
        print()
        
        # Compare each price directly to Black-Scholes using pytest.approx
        assert computed_price == pytest.approx(exact_price, abs=tol)
    
    


