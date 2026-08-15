# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 10:32:40 2026

@author: Nicola
"""

import os 
import sys

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join("\\".join(root)))

import numpy as np
from datetime import datetime, timedelta

from process_class import Heston93
from EuropeanOption_class import EuroPutAssetOrNothing
from MonteCarloEngine import MonteCarloEngine
from FourierDampingEngine import FourierDampingEngine


PARAMETERS = [
    # S0,K,r,v0,kappa,theta,eta,rho,T
    (100,100,0.05,0.04,1.0,0.04,0.20,-0.70,1.0),
    (100,80, 0.05,0.04,1.0,0.04,0.20,-0.70,1.0),
    (100,120,0.05,0.04,1.0,0.04,0.20,-0.70,1.0),
    (100,90,0.05,0.04,1.0,0.04,0.20,-0.70,1.0),
    (105,85, 0.05,0.04,1.0,0.04,0.20,-0.70,1.0),
    (110,110,0.05,0.04,1.0,0.04,0.20,-0.70,1.0)]


def test_heston_mc_vs_fourier():
    
    np.random.seed(42)

    for (S0,K,r,v0,kappa,theta,eta,rho,T) in PARAMETERS:

        today = datetime(2026,1,1)
        maturity = today + timedelta(days=int(365*T))

        option = EuroPutAssetOrNothing(K,maturity)

        process = Heston93(mu=r, kappa=kappa, theta=theta, eta=eta, rho=rho)

        mc_price = MonteCarloEngine(option, process, [S0,v0], r, n_steps=252, n=18, day=today)

        fourier_price = FourierDampingEngine(option, process, [S0,v0], r, alpha=-1.5, day=today)

        print()
        print("MC =",mc_price)
        print("Fourier =",fourier_price)
        print("Diff =",mc_price-fourier_price)

        np.testing.assert_allclose(mc_price, fourier_price, rtol=5e-1, atol=5e-1)

test_heston_mc_vs_fourier()