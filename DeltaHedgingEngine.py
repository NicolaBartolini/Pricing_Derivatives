# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:01:31 2026

@author: Nicola
"""


from PricingEngine_class import PricingEngine 
from process_class import process
from datetime import datetime 
from copy import copy 

def compute_Delta(option, process : process, pricing_engine : PricingEngine, X0, r : float, today:datetime, dx = 1e-4):
    
    X1 = copy(X0)
    X1[0] += dx 
    
    P0 = pricing_engine.evaluate_option(option, process, X0, r, today) 
    P1 = pricing_engine.evaluate_option(option, process, X1, r, today) 
    
    delta = (P1-P0)/dx 
    
    return delta
    