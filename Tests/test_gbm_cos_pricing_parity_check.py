# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:47:14 2026

@author: Nicola
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:00:32 2026

@author: Nicola
"""
import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))

import numpy as np
import pytest

from datetime import datetime, timedelta
from scipy.stats import norm

from process_class import GBM
from EuropeanOption_class import EuroCall, EuroPut
from CosPricingEngine import CosPricingEngine


def black_scholes_call(S0, K, r, sigma, T):
    d1 = (
        np.log(S0 / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return ( S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


@pytest.mark.parametrize("sigma", np.linspace(0.10, 0.50, 5))
@pytest.mark.parametrize("T", [0.25, 0.50, 0.75, 1.00])
def test_cos_matches_black_scholes(sigma, T):

    S0 = 100.0
    K = 100.0
    r = 0.05

    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=int(365 * T))

    option = EuroCall(K, maturity)
    option2 = EuroPut(K, maturity)
    process = GBM(r, sigma)

    call_price = CosPricingEngine(option=option, process=process, X0=[S0], r=r, N=256, L=10, day=today)
    put_price = CosPricingEngine(option=option2, process=process, X0=[S0], r=r, N=256, L=10, day=today)

    put_parity = call_price - S0 + K*np.exp(-r*T)
    
    # bs_price = black_scholes_call( S0, K, r, sigma, T)
    
    print(f"COS     call  = {call_price}")
    print(f"COS       put = {put_price}")
    print(f"Parity  price = {put_parity}")
    print()

    np.testing.assert_allclose( put_price, put_parity, rtol=1e-2, atol=1e-2)