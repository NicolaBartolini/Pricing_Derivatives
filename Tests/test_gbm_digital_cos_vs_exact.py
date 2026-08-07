# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:15:27 2026

@author: Nicola
"""

import sys
sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance")

import numpy as np
import pytest
from datetime import datetime, timedelta
from scipy.stats import norm

from process_class import GBM
from EuropeanOption_class import EuroCallDigital
from CosPricingEngine import CosPricingEngine


# -------------------------------------------------------
# Black-Scholes digital call
# -------------------------------------------------------

def bs_digital_call(S0, K, r, sigma, T):

    d2 = (
        np.log(S0/K)
        + (r - 0.5*sigma**2)*T
    )/(sigma*np.sqrt(T))

    return np.exp(-r*T)*norm.cdf(d2)


# -------------------------------------------------------
# Test parameters
# -------------------------------------------------------

PARAMETERS = [
    # S0, K, r, sigma, T
    (100, 100, 0.05, 0.20, 1.0),
    (100, 80, 0.05, 0.20, 1.0),
    (100, 120, 0.05, 0.20, 1.0),

    (100, 100, 0.01, 0.10, 0.5),
    (100, 100, 0.10, 0.30, 2.0),

    (120, 100, 0.05, 0.25, 1.5),
    (80, 100, 0.03, 0.15, 1.0),

    (100, 90, 0.00, 0.40, 0.25),
    (150, 100, 0.07, 0.35, 3.0),
    (50, 70, 0.02, 0.20, 2.0),
]


@pytest.mark.parametrize(
    "S0,K,r,sigma,T",
    PARAMETERS
)
def test_gbm_digital_cos_vs_exact(
    S0,
    K,
    r,
    sigma,
    T
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


    cos_price = CosPricingEngine(
        option=option,
        process=process,
        X0=[S0],
        r=r,
        N=256,
        L=10,
        day=today
    )


    bs_price = bs_digital_call(
        S0,
        K,
        r,
        sigma,
        T
    )


    print()
    print(f"S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"COS price = {cos_price}")
    print(f"BS  price = {bs_price}")
    print(f"Diff      = {cos_price-bs_price}")


    np.testing.assert_allclose(
        cos_price,
        bs_price,
        rtol=1e-2,
        atol=1e-2
    )