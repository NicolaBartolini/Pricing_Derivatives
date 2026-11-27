# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:56:32 2026

@author: Nicola
"""

from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime 

import numpy as np 

from instruments.conventions.DayCountConvetion_class import * 

from process.additive_process import additive_process 


class ZCB:
    """Risk-free zero-coupon-bond (zcb) class"""
    
    def __init__(self, maturity_date : datetime, convention : DayCountConvention = None, notional : float = 1.0):
        
        self.maturity_date = maturity_date 
        self.notional = notional 
        
        if convention==None:
            self.convention = Actual365() 
        else:
            self.convention = convention 
    
    def get_tenor(self, start_date : datetime) -> float :
        
        tenor = self.convention.year_fraction(start_date, self.maturity_date)
        
        return tenor 



class CB:
    """Risk-free coupon-bond (cb) class"""
    
    def __init__(self, coupon_rate : float, payment_schedule : list, maturity_date : datetime, convention : DayCountConvention = None, notional : float = 1.0):
        
        self.coupon_rate = coupon_rate 
        self.payment_schedule = payment_schedule # list of datetime with the dates of the coupon rates
        self.maturity_date = maturity_date 
        self.notional = notional 
        
        if convention==None:
            self.convention = Actual365() 
        else:
            self.convention = convention 
    
    def get_tenor(self, start_date : datetime) -> float :
        
        tenor = self.convention.year_fraction(start_date, self.maturity_date)
        
        return tenor 
    
    def get_tenor_coupon_payments(self, day : datetime):
        
        result = [] 
        
        for payment_day in self.payment_schedule:
            
            if payment_day>=day:
                
                tenor = self.convention.year_fraction(day, payment_day) 
                result.append(tenor)
        
        return result  


    
    
    
    