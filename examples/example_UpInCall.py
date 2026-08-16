# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:27:37 2026

@author: Nicola
"""

import numpy as np 
from datetime import datetime, timedelta 
from process_class import GBM 
from MonteCarloEngine import PathDependentMonteCarloEngine
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
    
    today = datetime(2026, 1, 4)
    maturity_date = today + timedelta(int(tenor*year_days))
    
    option = UpInCall(K, L, maturity_date) 
    process = GBM(r, sigma) 
    
    n = 10
    
    option_price = PathDependentMonteCarloEngine(option, process, [S0], r, nSteps, n, today)
    