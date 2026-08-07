# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:04:04 2026

@author: Nicola
"""

import numpy as np 
from copy import copy 
from datetime import datetime 

from EuropeanOption_class import EuropeanOption 
from process_class import process
from typing import Sequence


def CosPricingEngine(option : EuropeanOption, process : process, X0 : Sequence[float], r : float, N=32, day=None, L=10):
    
    if day==None:
        day = datetime.now() 
    
    T = option.get_tenor(day)
    
    S0 = X0[0]
    X1 = copy(X0)
    X1[0] = 1 
    
    c1, c2, c4 = process.cos_cumulants(X1, T) 
    
    # print(c1)
    # print(c2)
    # print(c4)
    # print()
        
    a = c1 - L*np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L*np.sqrt(c2 + np.sqrt(c4))
    
    
    bma = b-a
    k  = np.arange(N+1)
    u  = k * np.pi/(b-a)
    
    CF = process.characteristic_fun(u, X1, T) # computing the characteristic function 
    
    x  = np.log(S0/option.strike)
    Term = np.exp(1j * k * np.pi * (x-a)/bma)
    Fk = np.real(np.multiply(CF, Term))
    Fk[0] = 0.5 * Fk[0] 
        
    V_COS = option.strike * np.sum(np.multiply(Fk, option.UK(k, a, b))) * np.exp(-r*T)
    
    return V_COS
        
        
    