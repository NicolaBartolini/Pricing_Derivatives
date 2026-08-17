# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:50:42 2026

@author: Nicola
"""

import sys 

sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance") 

import numpy as np
from datetime import datetime,timedelta

from process_class import VarianceGamma
from EuropeanOption_class import EuroCall
from MonteCarloEngine import MonteCarloEngine
from CosPricingEngine import CosPricingEngine



PARAMETERS=[
    (0.05,0.2,0.2,-0.1),
    (0.03,0.15,0.3,-0.2),
    (0.07,0.25,0.1,0.05)]


def test_vg_mc_vs_cos():

    S0=100
    K=100
    r=0.05

    today=datetime(2026,1,1)
    maturity=today+timedelta(days=365)

    option=EuroCall(K,maturity)

    for params in PARAMETERS:

        process=VarianceGamma(*params)

        mc=MonteCarloEngine(option, process, [S0], r, 252, n=14, day=today)

        cos=CosPricingEngine(option, process, [S0], r, N=256, L=10, day=today)

        print(mc,cos)

        np.testing.assert_allclose(mc, cos, rtol=5e-2)