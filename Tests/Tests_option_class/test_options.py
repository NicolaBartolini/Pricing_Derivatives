# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:48:35 2026

@author: Nicola
"""

import numpy as np 
import pytest 
from datetime import datetime, timedelta

from instruments.options.EuropeanOption_class import * 


class TestOptionClass:
    
    def test_object_creation(self):
        
        strike = 100 
        S0 = 100 
        
        date = datetime(2026, 11, 5)
                
        option1 = EuroCall(strike, date)
        option2 = EuroPut(strike, date)
        
        assert isinstance(option1, EuroCall)
        assert isinstance(option2, EuroPut)
    
    def test_moneyness_eurocall(self):
        
        strike = 100 
        S0 = 99 
        
        date = datetime(2026, 11, 5)
                
        option = EuroCall(strike, date)
        
        assert not option.moneyness(S0)
    
    def test_moneyness_europut(self):
        
        strike = 100 
        S0 = 99 
        
        date = datetime(2026, 11, 5)
                
        option = EuroPut(strike, date)
        
        assert option.moneyness(S0)
    
    def test_moneyness_eurocalldigital(self):
        
        strike = 100 
        S0 = 99 
        
        date = datetime(2026, 11, 5)
                
        option = EuroCallDigital(strike, date)
        
        assert not option.moneyness(S0)
    
    def test_moneyness_europutdigital(self):
        
        strike = 100 
        S0 = 99 
        
        date = datetime(2026, 11, 5)
                
        option = EuroPutDigital(strike, date)
        
        assert option.moneyness(S0)
        
    def test_moneyness_eurocallAoN(self):
        
        strike = 100 
        S0 = 99 
        
        date = datetime(2026, 11, 5)
        
        option = EuroCallAssetOrNothing(strike, date)
        
        assert not option.moneyness(S0)
    
    def test_moneyness_europutAoN(self):
        
        strike = 100 
        S0 = 99 
        
        date = datetime(2026, 11, 5)
        
        option = EuroPutAssetOrNothing(strike, date)
        
        assert option.moneyness(S0)
    
    
    
    def test_moneyness_capl(self):
        
        strike = .03 
        r0 = .02
        
        date = datetime(2026, 11, 5)
                
        option = Capl(strike, date, 1)
        
        assert not option.moneyness(r0)
    
    def test_moneyness_floorl(self):
        
        strike = .03 
        r0 = .02
        
        date = datetime(2026, 11, 5)
                
        option = Floorl(strike, date, 1)
        
        assert option.moneyness(r0)
        
        
        