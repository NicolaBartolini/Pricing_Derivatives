# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:10:16 2026

@author: Nicola
"""

from abc import ABC, abstractmethod 
from MonteCarloEngine import MonteCarloEngine, PathDependentMonteCarloEngine
from FourierDampingEngine import FourierDampingEngine 
from LongstaffSchwartzEngine import LongstaffSchwartzEngine 
from CosPricingEngine import CosPricingEngine

class PricingEngine(ABC):
    
    @abstractmethod 
    def evaluate_option(self, option, process, X0, r, day):
        pass 
    

class MonteCarloPricer(PricingEngine):
    
    def __init__(self, n_steps=1, n=10):
        
        self.n_steps = n_steps
        self.n = n 
        
    def evaluate_option(self, option, process, X0, r, day, path_dependence=False):
        
        if path_dependence:
            
            price = PathDependentMonteCarloEngine(option, process, X0, r, self.n_steps, self.n, day)
            
            return price
            
        price = MonteCarloEngine(option, process, X0, r, self.n_steps, self.n, day) 
        
        return price


class FourierDampingPricer(PricingEngine):
    
    def __init__(self, alpha : float, a=0, b=250, n=5000):
        
        self.alpha = alpha 
        self.a = a 
        self.b = b 
        self.n = n 
    
    def evaluate_option(self, option, process, X0, r, day):
        
        price = FourierDampingEngine(option, process, X0, r, self.alpha, day, self.a, self.b, self.n)
        
        return price 

class CosPricer(PricingEngine):
    
    def __init__(self, N=32, L=10):
        
        self.N = 32 
        self.L = L 
    
    def evaluate_option(self, option, process, X0, r, day):
        
        price = CosPricingEngine(option, process, X0, r, self.N, day, self.L)
        
        return price 

class LongstaffSchwartzPricer(PricingEngine):
    
    def __init__(self, N=10, degree=3, basis='Laguerre'):
        
        self.N = N 
        self.basis = basis 
        self.degree = degree 
    
    def evaluate_option(self, american_option, process, X0, r, day):
        
        price = LongstaffSchwartzEngine(process, american_option, X0, day, r, self.N, self.degree, self.basis)
        
        return price 
    
    
    