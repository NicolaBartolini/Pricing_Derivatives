# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:02:21 2026

@author: Nicola
"""

import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))

import pytest 

import numpy as np
from scipy.stats import norm 
from datetime import datetime, timedelta
from MonteCarloEngine import PathDependentMonteCarloEngine
from AsianOption_class import AsianCall 
from process_class import GBM



def geometric_asian_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Kemna-Vorst closed-form price for a continuously
    sampled geometric-average Asian call option.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility of the underlying.
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        Asian call price.
    """

    if S0 <= 0:
        raise ValueError("S0 must be positive.")

    if K <= 0:
        raise ValueError("K must be positive.")

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if T <= 0:
        return max(S0 - K, 0.0)

    # Effective volatility of the geometric average
    sigma_g = sigma / np.sqrt(3.0)

    # Forward price of the geometric average
    F_g = S0 * np.exp((r - sigma**2 / 6.0) * T)

    # Standard Black-Scholes quantities
    d1 = (np.log(F_g / K) + 0.5 * sigma_g**2 * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    # Discounted Black-Scholes price
    price = np.exp(-r * T) * (F_g * norm.cdf(d1) - K * norm.cdf(d2))

    return price


def test_asian_option():
    
    np.random.seed(42)
    
    S0 = 1 
    sigma = .2 
    r = 0.02
    
    process = GBM(mu=r, sigma=sigma)
    
    X0 = [S0]
    
    n_steps = 252
    n_paths = 20
    
    # 1 year maturity option
    today = datetime(2026,1,1)
    maturity = today + timedelta(days=int(365))
    
    STRIKES = [.8, .9, 1, 1.1, 1.2]
    
    for K in STRIKES:
        
        exact_price = geometric_asian_call(S0, K, r, sigma, 1)
        
        option = AsianCall(K, maturity, 'geometric')
        
        mc_price = PathDependentMonteCarloEngine(option, process, X0, r, n_steps, n_paths)
        
        print()
        print("MC    =",mc_price)
        print("Exact =",exact_price)
        print("Diff  =",mc_price-exact_price)

        np.testing.assert_allclose(mc_price, exact_price, rtol=5e-2, atol=5e-2)
    
    

# test_asian_option()
