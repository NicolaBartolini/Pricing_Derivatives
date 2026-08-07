# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 09:41:00 2026

@author: Nicola
"""

import numpy as np 
from datetime import datetime, timedelta 
from abc import ABC, abstractmethod 


###### Utils 

def business_days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Returns the number of working days (Monday-Friday)
    between two dates, inclusive.

    Parameters
    ----------
    start_date : datetime
    end_date   : datetime

    Returns
    -------
    int
        Number of business days.
    """

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    current = start_date.date()
    end = end_date.date()

    business_days = 0

    while current <= end:
        if current.weekday() < 5:      # Monday=0, ..., Friday=4
            business_days += 1
        current += timedelta(days=1)

    return business_days



class AmericanOption(ABC):
    
    @abstractmethod 
    def payoff(self, S : float) -> float:
        """payoff function"""
        pass 
    
    @abstractmethod 
    def get_tenor(self, date : datetime) -> float:
        pass 
    
    @abstractmethod 
    def get_business_number_of_days(self, start_date: datetime) -> int:  
        pass 
    
    @abstractmethod
    def get_payoff_grid(self, start_date : datetime) -> np.array:
        """
        this function returns the array with the time points from 0 to T (the tenor)
        example: [0, .5, 1]
        """
        pass 


class AmericanCall(AmericanOption): 
    
    def __init__(self, strike : float, maturity_date : datetime) : 
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        
    def get_tenor(self, date):
        
        diff = self.maturity_date - date
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S):
        
        return np.maximum(S-self.strike, 0) 
    
    def get_business_number_of_days(self, start_date):
        
        return business_days_between(start_date, self.maturity_date)
    
    def get_payoff_grid(self, start_date):
        
        n_days = business_days_between(start_date, self.maturity_date)
        
        T = self.get_tenor(start_date)
        
        grid = np.linspace(0, T, n_days+1) 
        
        return grid
    
    
class AmericanPut(AmericanOption): 
    
    def __init__(self, strike : float, maturity_date : datetime) : 
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        
    def get_tenor(self, date):
        
        diff = self.maturity_date - date
        
        tenor = diff.days/365 
        
        return tenor  
    
    def get_business_number_of_days(self, start_date):
        
        return business_days_between(start_date, self.maturity_date)
    
    def payoff(self, S):
        
        return np.maximum(self.strike - S, 0)
    
    def get_payoff_grid(self, start_date):
        
        n_days = business_days_between(start_date, self.maturity_date)
        
        T = self.get_tenor(start_date)
        
        grid = np.linspace(0, T, n_days+1) 
        
        return grid
