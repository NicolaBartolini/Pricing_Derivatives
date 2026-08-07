# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:29:09 2026

@author: Nicola
"""

import sys
sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance")

import numpy as np
import pytest

from datetime import datetime, timedelta
from scipy.stats import norm

from process_class import GBM
from EuropeanOption_class import EuroCall
from FourierDampingEngine import FourierDampingEngine


def black_scholes_call(S0, K, r, sigma, T):
    d1 = (
        np.log(S0 / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return (
        S0 * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


@pytest.mark.parametrize("sigma", np.linspace(0.10, 0.50, 5))
@pytest.mark.parametrize("T", [0.25, 0.50, 0.75, 1.00])
def test_fourier_engine_matches_black_scholes(sigma, T):

    S0 = 100.0
    K = 100.0
    r = 0.05
    alpha = 1.5

    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=int(365 * T))

    option = EuroCall(K, maturity)
    process = GBM(r, sigma)

    fourier_price = FourierDampingEngine( option=option, process=process, X0=[S0], r=r, 
                                         alpha=alpha, day=today)

    bs_price = black_scholes_call(S0, K, r, sigma, T)

    np.testing.assert_allclose(fourier_price, bs_price, rtol=1e-2, atol=1e-3)