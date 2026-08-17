# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:25:39 2026

@author: Nicola
"""

# test_heston_fourier_vs_cos_digital.py

import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))


from datetime import datetime,timedelta
import numpy as np


from process_class import Heston93
from EuropeanOption_class import EuroCallDigital
from FourierDampingEngine import FourierDampingEngine
from CosPricingEngine import CosPricingEngine



PARAMETERS=[

(100,100,0.05,0.04,1.0,0.04,0.20,-0.70,1.0),
(100,80,0.05,0.04,1.0,0.04,0.20,-0.70,1.0),
(100,120,0.05,0.04,1.0,0.04,0.20,-0.70,1.0)

]



def test_heston_fourier_vs_cos():

    for (S0,K,r,v0,kappa,theta,eta,rho,T) in PARAMETERS:

        today=datetime(2026,1,1)

        maturity=today+timedelta(days=int(365*T))

        option=EuroCallDigital(K,maturity)

        process=Heston93(mu=r, kappa=kappa, theta=theta, eta=eta, rho=rho)

        fourier_price=FourierDampingEngine(option, process, [S0,v0], r, alpha=1.5, day=today)

        cos_price=CosPricingEngine(option, process, [S0,v0], r, N=512, L=10, day=today)

        print()
        print("Fourier =",fourier_price)
        print("COS =",cos_price)
        print("Diff =",fourier_price-cos_price)

        np.testing.assert_allclose(fourier_price, cos_price, rtol=1e-4, atol=1e-4)