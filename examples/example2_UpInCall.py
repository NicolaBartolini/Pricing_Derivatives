# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 08:48:13 2026

@author: Nicola
"""


import numpy as np 
from datetime import datetime, timedelta 
from process_class import GBM 
from PricingEngine_class import MonteCarloPricer
from BarrierOption_class import UpInCall


if __name__=='__main__':
    
    np.random.seed(42)
    
    S0 = 1 
    K = 1.15 
    L = 1.075 
    
    sigma = .2
    r = .025
    
    tenor = .5
    year_days = 365
    nSteps = int(year_days * tenor)
    
    n = 10
    
    today = datetime(2026, 1, 4)
    maturity_date = today + timedelta(int(tenor*year_days))
    
    option = UpInCall(K, L, maturity_date) 
    process = GBM(r, sigma) 
    pricer = MonteCarloPricer(nSteps, n)
        
    option_price = pricer.evaluate_option(option, process, [S0], r, today, True)
    