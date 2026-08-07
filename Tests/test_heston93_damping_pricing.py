# -*- coding: utf-8 -*-
"""
Tests for Fourier damping pricing of Heston93
"""

import sys
sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance")

import numpy as np
import pytest

from datetime import datetime, timedelta

from process_class import Heston93
from EuropeanOption_class import EuroCall, EuroPut
from FourierDampingEngine import FourierDampingEngine
from MonteCarloEngine import MonteCarloEngine


np.random.seed(42)


# -----------------------------------------------------------
# Different Heston parameter configurations
# -----------------------------------------------------------
# v0, rho, eta, theta, kappa
PARAMETER_SETS = [
    (0.04, -0.30, 0.20, 0.04, 1.0),
    (0.06, -0.70, 0.40, 0.05, 2.0),
    (0.09,  0.20, 0.30, 0.09, 1.5),
]


STRIKES = [80.0, 100.0, 120.0]



# -----------------------------------------------------------
# CALL TEST
# -----------------------------------------------------------

@pytest.mark.parametrize(
    "v0,rho,eta,theta,kappa",
    PARAMETER_SETS
)
@pytest.mark.parametrize(
    "K",
    STRIKES
)
def test_heston_call_fourier_vs_mc(
        v0,
        rho,
        eta,
        theta,
        kappa,
        K
):

    S0 = 100.0
    r = 0.05

    today = datetime(2026,1,1)
    maturity = today + timedelta(days=365)


    option = EuroCall(
        K,
        maturity
    )


    process = Heston93(
        mu=r,
        kappa=kappa,
        theta=theta,
        eta=eta,
        rho=rho
    )


    # Fourier damping
    fourier_price = FourierDampingEngine(
        option=option,
        process=process,
        X0=[S0,v0],
        r=r,
        alpha=1.5,
        day=today,
        b=250,
        n=5000
    )


    # Monte Carlo
    mc_price = MonteCarloEngine(
        option=option,
        process=process,
        X0=[S0,v0],
        r=r,
        n_steps=252,
        n=16,
        day=today
    )


    print("\nCALL")
    print("K =",K)
    print("Fourier :", fourier_price)
    print("MC      :", mc_price)


    np.testing.assert_allclose(
        mc_price,
        fourier_price,
        rtol=5e-2,
        atol=5e-2
    )



# -----------------------------------------------------------
# PUT TEST
# -----------------------------------------------------------

@pytest.mark.parametrize(
    "v0,rho,eta,theta,kappa",
    PARAMETER_SETS
)
@pytest.mark.parametrize(
    "K",
    STRIKES
)
def test_heston_put_fourier_vs_mc(
        v0,
        rho,
        eta,
        theta,
        kappa,
        K
):

    S0 = 100.0
    r = 0.05

    today = datetime(2026,1,1)
    maturity = today + timedelta(days=365)


    option = EuroPut(
        K,
        maturity
    )


    process = Heston93(
        mu=r,
        kappa=kappa,
        theta=theta,
        eta=eta,
        rho=rho
    )


    # Fourier damping for put
    # same integrand, negative alpha
    fourier_price = FourierDampingEngine(
        option=option,
        process=process,
        X0=[S0,v0],
        r=r,
        alpha=-1.5,
        day=today,
        b=250,
        n=5000
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


    print("\nPUT")
    print("K =",K)
    print("Fourier :", fourier_price)
    print("MC      :", mc_price)


    np.testing.assert_allclose(
        mc_price,
        fourier_price,
        rtol=5e-2,
        atol=5e-2
    )