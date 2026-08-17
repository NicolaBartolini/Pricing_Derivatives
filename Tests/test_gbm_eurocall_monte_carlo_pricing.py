# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:52:37 2026

@author: Nicola
"""
import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))

import numpy as np
from scipy.stats import norm 

import pytest

from datetime import datetime, timedelta


from process_class import GBM 
from EuropeanOption_class import EuroCall 
from MonteCarloEngine import MonteCarloEngine

np.random.seed(42)

def black_scholes_call(S0, K, r, sigma, T):

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)



@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30])
@pytest.mark.parametrize("T", [0.25, 0.50, 1.00])
def test_mc_matches_black_scholes(sigma, T):

    S0 = 100.0
    K = 100.0
    r = 0.05

    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=int(365 * T))

    option = EuroCall(K, maturity)
    process = GBM(r, sigma)
    
    n = 18

    mc_price = MonteCarloEngine(option, process, X0=[S0], r=r, n_steps=100, n=n, day=today) 
    
    bs_price = black_scholes_call(S0, K, r, sigma, T)

    assert np.isclose(mc_price, bs_price, rtol=2e-2) 
    
    