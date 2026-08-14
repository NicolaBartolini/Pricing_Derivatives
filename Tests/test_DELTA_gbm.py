# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:22:08 2026

@author: Nicola
"""
import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join("\\".join(root)))

from process_class import GBM 
from EuropeanOption_class import EuroCall 
from PricingEngine_class import CosPricer
from DeltaHedgingEngine import compute_Delta

import pytest 

from datetime import datetime, timedelta

import numpy as np 
from scipy.stats import norm

def black_scholes_call_delta(S0: float | np.ndarray, K: float | np.ndarray, r: float, sigma: float, T: float,) -> float | np.ndarray:
    """Calculates the Black-Scholes Delta for a European Call option.

    Parameters
    ----------
    S0 : float or np.ndarray
        Current underlying asset price.
    K : float or np.ndarray
        Strike price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Volatility of the underlying asset (annualized).
    T : float
        Time to maturity in years.

    Returns
    -------
    float or np.ndarray
        Delta of the call option (bounded between 0 and 1).
    """
    if T <= 0:
        # At expiration, Delta is 1 if ITM, 0 if OTM
        return np.where(S0 > K, 1.0, 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return norm.cdf(d1)

@pytest.mark.parametrize("sigma", np.linspace(0.10, 0.50, 5))
@pytest.mark.parametrize("T", [0.25, 0.50, 0.75, 1.00])
def test_DELTA(sigma, T):
    
    np.random.seed(42)
    
    S0 = 100 
    # sigma = .2 
    r = .02 
    
    model = GBM(r, sigma)
    
    strike = 100 # option strike 
    # T = 1 # option tenor 
    
    current_day = datetime(2026, 3, 1)
    maturity_date = current_day + timedelta(int(T*365))
    
    option = EuroCall(strike, maturity_date)
    
    pricer = CosPricer()
    
    delta = compute_Delta(option, model, pricer, [S0], r, current_day, 1e-6)
    
    exact_delta = black_scholes_call_delta(S0, strike, r, sigma, T) 
    
    diff = exact_delta - delta
    
    assert abs(diff) < 5e-4