# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:50:50 2026

@author: Nicola
"""

import sys 

sys.path.append(r"C:\Users\Nicola\Documents\Python_Scripts\0_finance") 

import numpy as np
from datetime import datetime,timedelta


from process_class import VarianceGamma
from EuropeanOption_class import EuroCall
from FourierDampingEngine import FourierDampingEngine
from CosPricingEngine import CosPricingEngine



PARAMETERS=[
    (0.05,0.2,0.2,-0.1),
    (0.03,0.15,0.3,-0.2),
    (0.07,0.25,0.1,0.05)]


def test_vg_damping_vs_cos():

    S0=100
    K=100
    r=0.05
    alpha=1.5


    today=datetime(2026,1,1)
    maturity=today+timedelta(days=365)

    option=EuroCall(K,maturity)

    for params in PARAMETERS:

        process=VarianceGamma(*params)

        fd=FourierDampingEngine(option, process, [S0], r, alpha, day=today)

        cos=CosPricingEngine(option, process, [S0], r, N=256, L=10,day=today)

        print(fd,cos)

        np.testing.assert_allclose(fd, cos, rtol=1e-5, atol=1e-5)