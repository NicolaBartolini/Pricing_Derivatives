# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:54:51 2026

@author: Nicola
"""


import sys 

sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance")

import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm

from process_class import GBM
from EuropeanOption_class import EuroCallDigital
from FourierDampingEngine import FourierDampingEngine

import pytest

def bs_digital_call(S0, K, r, sigma, T):

    d2 = (
        np.log(S0/K)
        + (r - 0.5*sigma**2)*T
    )/(sigma*np.sqrt(T))

    return np.exp(-r*T)*norm.cdf(d2)


# -------------------------------------------------------
# Parameter sets:
# S0, K, r, sigma, T, alpha
# -------------------------------------------------------

PARAMETERS = [

    (100, 100, 0.05, 0.20, 1.0, 1.5),   # ATM
    (100,  80, 0.05, 0.20, 1.0, 1.5),   # ITM
    (100, 120, 0.05, 0.20, 1.0, 1.5),   # OTM

    (100, 100, 0.01, 0.10, 0.5, 1.5),   # low vol short maturity
    (100, 100, 0.10, 0.40, 2.0, 1.5),   # high vol long maturity

    (150, 100, 0.05, 0.25, 1.0, 1.5),   # deep ITM
    (50,  100, 0.05, 0.25, 1.0, 1.5),   # deep OTM

    (100, 100, -0.01, 0.20, 1.0, 1.5),  # negative rates

    (120, 110, 0.03, 0.15, 3.0, 1.5),   # long maturity

    (80,  90, 0.07, 0.35, 0.25, 1.5),   # short maturity
]


@pytest.mark.parametrize(
    "S0,K,r,sigma,T,alpha",
    PARAMETERS
)
def test_gbm_digital_fourier(
    S0,
    K,
    r,
    sigma,
    T,
    alpha
):

    today = datetime(2026,1,1)
    maturity = today + timedelta(days=int(365*T))

    option = EuroCallDigital(
        K,
        maturity
    )

    process = GBM(
        r,
        sigma
    )

    fourier_price = FourierDampingEngine(
        option,
        process,
        [S0],
        r,
        alpha,
        day=today
    )

    bs_price = bs_digital_call(
        S0,
        K,
        r,
        sigma,
        T
    )

    print("\n---------------------------")
    print(f"S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"Fourier = {fourier_price}")
    print(f"BS      = {bs_price}")
    print(f"Error   = {fourier_price-bs_price}")

    np.testing.assert_allclose(
        fourier_price,
        bs_price,
        rtol=1e-2,
        atol=1e-2
    )