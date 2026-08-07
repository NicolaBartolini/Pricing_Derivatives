# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:38:48 2026

@author: Nicola
"""

import numpy as np 
from datetime import datetime

def MonteCarloEngine(option, process, X0, r, n_steps, n=10, day=None):
    
    if day==None:
        day = datetime.now()
    
    T = option.get_tenor(day)
    
    sim = process.simulate( X0, T, n_steps, n)
    
    if isinstance(sim, tuple):
        ST = sim[0][-1]      # Heston: (S, V)
    else:
        ST = sim[-1]         # GBM: S
    
    price = np.mean(option.payoff(ST)) * np.exp(-r*T)
    
    return price