# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:01:31 2026

@author: Nicola
"""


import numpy as np 
from PricingEngine_class import PricingEngine 
from process_class import process
from datetime import datetime, timedelta 
from copy import copy 

def compute_Delta(option, process : process, pricing_engine : PricingEngine, X0, r : float, today:datetime, dx = 1e-4):
    
    X1 = copy(X0)
    X1[0] += dx 
    
    P0 = pricing_engine.evaluate_option(option, process, X0, r, today) 
    P1 = pricing_engine.evaluate_option(option, process, X1, r, today) 
    
    delta = (P1-P0)/dx 
    
    return delta
    
if __name__=='__main__':
    
    from process_class import GBM 
    from EuropeanOption_class import EuroCall 
    from PricingEngine_class import CosPricer
    
    np.random.seed(42)
    
    S0 = 100 
    sigma = .2 
    r = .02 
    
    model = GBM(r, sigma)
    
    strike = 100 # option strike 
    T = 1 # option tenor 
    
    current_day = datetime(2026, 3, 1)
    maturity_date = current_day + timedelta(int(T*365))
    
    option = EuroCall(strike, maturity_date)
    
    pricer = CosPricer()
    
    delta0 = compute_Delta(option, model, pricer, [S0], r, current_day, 1e-3)
    delta1 = compute_Delta(option, model, pricer, [S0], r, current_day, 1e-4)
    delta2 = compute_Delta(option, model, pricer, [S0], r, current_day, 1e-5)
    delta3 = compute_Delta(option, model, pricer, [S0], r, current_day, 1e-6)

    
    print(f"Option delta : {delta0}")
    print(f"Option delta : {delta1}")
    print(f"Option delta : {delta2}")
    print(f"Option delta : {delta3}")