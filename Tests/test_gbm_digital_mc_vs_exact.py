# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:43:33 2026

@author: Nicola
"""

# test_gbm_digital_mc_pricing.py

import sys 

sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance")

import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm

from process_class import GBM
from EuropeanOption_class import EuroCallDigital
from MonteCarloEngine import MonteCarloEngine


np.random.seed(42)


def bs_digital_call(S0, K, r, sigma, T):

    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return np.exp(-r*T) * norm.cdf(d2)


def test_gbm_digital_mc():

    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0

    today = datetime(2026,1,1)
    maturity = today + timedelta(days=365)

    option = EuroCallDigital(K, maturity)
    process = GBM(r, sigma)

    mc_price = MonteCarloEngine(
        option,
        process,
        [S0],
        r,
        n_steps=252,
        n=18,
        day=today
    )

    bs_price = bs_digital_call(S0,K,r,sigma,T)

    print(f"MC  price = {mc_price}")
    print(f"BS  price = {bs_price}")

    np.testing.assert_allclose(
        mc_price,
        bs_price,
        rtol=2e-2,
        atol=2e-2
    )