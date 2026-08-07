# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:07:08 2026

@author: Nicola
"""

from FitEngine import fit_process 
from process_class import VarianceGamma 
from EuropeanOption_class import EuroCall, EuroPut 
from datetime import datetime 
from CosPricingEngine import CosPricingEngine

import numpy as np  



params = [0.03, 0.25,0.1,0.05] 

maturities = [datetime(2026, 9, 30), datetime(2026, 10, 30), datetime(2026, 11, 30)]

process=VarianceGamma(*params) 

S0 = 1 
STRIKES = [.8, .9, 1, 1.1, 1.2] 

X0 = [S0]
r = .03

today = datetime.now()

sim_options = []

for maturity in maturities:

    for strike in STRIKES:

        ############################################################
        # Call
        ############################################################

        call = EuroCall(strike, maturity)

        price = CosPricingEngine(option=call, process=process, X0=X0, r=r, day=today)

        call.set_market_premium(price)

        sim_options.append(call)

### Fit test 


initial_guess = np.random.uniform(1e-3, .6, 3) 
process_guess = VarianceGamma(*np.hstack((np.array(r), *initial_guess)).tolist())

res = fit_process(params=initial_guess, process=process_guess, OPTIONS=sim_options,
                  X0=X0, r=r, date=today, lb=1e-4, ub=1)


new_OPTIONS = []  

today = datetime.now()

for maturity in maturities:

    for strike in STRIKES:

        ############################################################
        # Call
        ############################################################

        call = EuroCall(strike, maturity)

        price = CosPricingEngine(option=call, process=process_guess, X0=X0, r=r, day=today)

        call.set_market_premium(price)

        new_OPTIONS.append(call)
        

for i in range(len(sim_options)):
    
    p1 = sim_options[i].get_premium()
    p2 = new_OPTIONS[i].get_premium() 
    print(f"{p1} \t:\t {p2} \t:\t {abs(p1-p2)}")