# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:46:58 2026

@author: Nicola
"""
import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))

import numpy as np
from datetime import datetime,timedelta

from process_class import VarianceGamma
from EuropeanOption_class import EuroCall
from MonteCarloEngine import MonteCarloEngine
from FourierDampingEngine import FourierDampingEngine


PARAMETERS=[
    (0.05,0.2,0.2,-0.1),
    (0.03,0.15,0.3,-0.2),
    (0.07,0.25,0.1,0.05)
]


def test_vg_mc_vs_damping():

    S0=100
    K=100
    r=0.05
    alpha=1.5

    today=datetime(2026,1,1)
    maturity=today+timedelta(days=365)

    option=EuroCall(K,maturity)

    for mu,sigma,nu,theta in PARAMETERS:
        process=VarianceGamma(mu, sigma, nu, theta)

        mc=MonteCarloEngine(option, process, [S0], r, 252, n=14, day=today)

        fd=FourierDampingEngine(option, process, [S0], r, alpha, day=today)

        print(mu,sigma,nu,theta)
        print(mc,fd)

        np.testing.assert_allclose(mc, fd, rtol=5e-2)