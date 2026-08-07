# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:23:59 2026

@author: Nicola
"""

import sys
sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance")

import numpy as np
import pytest

from datetime import datetime, timedelta

from process_class import Heston93
from EuropeanOption_class import EuroCall, EuroPut
from CosPricingEngine import CosPricingEngine 
from MonteCarloEngine import MonteCarloEngine

np.random.seed(42)

# ---------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------

PARAMETER_SETS = [
    # v0, rho, eta, theta, kappa
    (0.04, -0.30, 0.20, 0.04, 1.0)
]

# MATURITIES = [0.25, 0.50, 1.00]

STRIKES = [80.0, 100.0, 120.0]

@pytest.mark.parametrize("v0,rho,eta,theta,kappa", PARAMETER_SETS)
@pytest.mark.parametrize("K", STRIKES)
# @pytest.mark.parametrize("OptionClass", [EuroCall, EuroPut])
def test_heston_mc_vs_cos(
    v0,
    rho,
    eta,
    theta,
    kappa,
    K
    # OptionClass,
):

    S0 = 100.0
    r = 0.05

    today = datetime(2026,1,1)
    maturity = today + timedelta(days=365)

    # option = OptionClass(K, maturity)
    option = EuroCall(K, maturity)

    process = Heston93(
        mu=r,
        kappa=kappa,
        theta=theta,
        eta=eta,
        rho=rho
    )

    mc_price = MonteCarloEngine(
        option=option,
        process=process,
        X0=[S0,v0],
        r=r,
        n_steps=252,
        n=16,
        day=today
    )

    cos_price = CosPricingEngine(
        option=option,
        process=process,
        X0=[S0,v0],
        r=r,
        N=256,
        L=10,
        day=today
    )
    
    print(f"COS price : {cos_price}")
    print(f"MC  price : {mc_price}")
    print()
    
    np.testing.assert_allclose(
        mc_price,
        cos_price,
        rtol=2e-2,
        atol=2e-2
    )