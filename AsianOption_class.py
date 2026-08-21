# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:24:24 2026

@author: Nicola
"""

from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from scipy.stats.mstats import gmean
import numpy as np 


##################################

class AsianOption(ABC):
    
    def PathPependence():
        return True
    
    @abstractmethod 
    def set_market_premium(self, P : float) -> None:
        pass 
    
    @abstractproperty 
    def get_premium(self) -> float :
        pass
    
    @abstractmethod 
    def get_tenor(self, date : datetime):
        pass 
    
    @abstractmethod 
    def payoff(self, S : np.array):
        pass
    

class AsianCall(AsianOption):
    
    def __init__(self, strike : float, maturity_date : datetime, method : str):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        
        self.market_premium = np.nan 
        self.method = method
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium
    
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S):
        
        if self.method.lower()=='arithmetic':
        
            ST = np.mean(S, axis=0)
        
        elif self.method.lower() == 'geometric':
            
            ST = gmean(S, axis=0)
        
        else:
            raise ValueError('Only arithmetic or geometric average')
        
        return np.maximum(ST-self.strike, 0)


class AsianPut(AsianOption):
    
    def __init__(self, strike : float, maturity_date : datetime, method : str):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        
        self.market_premium = np.nan 
        self.method = method

        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium
    
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S):
        
        if self.method.lower()=='arithmetic':
        
            ST = np.mean(S, axis=0)
        
        elif self.method.lower() == 'geometric':
            
            ST = gmean(S, axis=0)
        
        else:
            raise ValueError('Only arithmetic or geometric average')
        
        return np.maximum(self.strike - ST, 0)
        
        
        
        