# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:54:58 2026

@author: Nicola
"""

import numpy as np 
from PricingEngine_class import PricingEngine 
from process_class import process
from datetime import datetime, timedelta

def compute_VaR(q : float, current_price:float, option, process : process, pricing_engine : PricingEngine, X0, r : float, today:datetime, VaR_date : datetime, n:int):
            
    # 1.a : determine the number of steps / days for the simulations
    
    N = 1<<n # number of Monte-Carlo simulations
    n_steps = (VaR_date - today).days
    t1 = n_steps/365
    # T = option.get_tenor(day)
    
    # Simulate the process underying up to the VaR evaluation date
    sim = process.simulate( X0, t1, n_steps, n)
    
    # Store the option price at t1
    option_prices_at_t1 = np.empty(N)
    
    if isinstance(sim, tuple):
        St1 = sim[0][-1]      # Heston - Bates: (S, V)
        nut1 = sim[1][-1]
        
        for i in range(0, N):
            
            X1 = [St1[i], nut1[i]]
            option_prices_at_t1[i] = pricing_engine.evaluate_option(option, process, X1, r, VaR_date)
        
    else:
        St1 = sim[-1]         # GBM - Merton - Variance GAmma: S 
        
        for i in range(0, N):
            
            X1 = [St1[i]]
            option_prices_at_t1[i] = pricing_engine.evaluate_option(option, process, X1, r, VaR_date)
    
    Loss = current_price - option_prices_at_t1
    
    VaR = np.quantile(Loss, q)
    
    return VaR, Loss
    

    
if __name__=='__main__':
    
    from process_class import GBM 
    from EuropeanOption_class import EuroCall 
    from PricingEngine_class import CosPricer
    import matplotlib.pyplot as plt
    
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
    
    current_price = pricer.evaluate_option(option, model, [S0], r, current_day)
    
    # VaR Part
    
    q = .95
    t1 = .5 
    VaR_date = current_day + timedelta(int(t1*365))
    n = 10 
    
    VaR, loss = compute_VaR(q, current_price, option, model, pricer, [S0], r, current_day, VaR_date, n)
    
    print(f"VaR           : {VaR}")
    print(f"Current price : {current_price}")
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(loss, density=True, bins=50)
    plt.grid()
    plt.show()