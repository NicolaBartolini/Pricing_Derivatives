# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:37:35 2026

@author: Nicola
"""

from TermStructure.TermStructure import *
from datetime import datetime, timedelta

import numpy as np 
import pytest

### Test discount factor class

class TestDiscountFactor: 
    
    def test_init_discount_factor(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        DF1 = DiscountFactor(t, T1, .85)
        
        assert isinstance(DF1, DiscountFactor)
        
    def test_monotonicity(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85)
        
        assert (DF1<DF2)

# Test discount factor auxiliary functions


class TestFunctions:
    
    def test_forward_1(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, 1.)
        DF2 = DiscountFactor(t, T2, 1.) 
        
        assert ComputeForwardDiscountValue(DF1, DF2) == 1 
        
    def test_forward_2(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85) 
        
        np.testing.assert_allclose(ComputeForwardDiscountValue(DF1, DF2), .89474, rtol=1e-2, atol=1e-3)
    
    def test_forward_3(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85) 
        
        DF_forward12 = DiscountFactor(t + timedelta(days=365*2), T2, ComputeForwardDiscountValue(DF1, DF2)) 
        
        f_rate = DF_forward12.GetRate() 
        
        np.testing.assert_allclose(f_rate, 0.1112, rtol=1e-2, atol=1e-4) 
    
    def test_forward_4(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85) 
        
        DF_forward12 = DiscountFactor(t + timedelta(days=365*2), T2, ComputeForwardDiscountValue(DF1, DF2))  
        
        DF12 = ComputeForwardDiscount(DF1, DF2) 
        
        assert isinstance(DF12, DiscountFactor)

# Test term structure class

class TestTermStructure:
    
    def test_init_term_structure(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85)
        
        l = [DF2, DF1] 
        
        TS = TermStructure(l)
        
        assert TS.discount_factors[0] < TS.discount_factors[1] 
        
    def test_get_discount_factor(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85)
        
        l = [DF2, DF1] 
        
        TS = TermStructure(l)
        
        assert TS.GetFactor(T1).t == DF1.t 
        assert TS.GetFactor(T1).T == DF1.T
        assert TS.GetFactor(T1).p == DF1.p
        
        assert TS.GetFactor(T2).t == DF2.t 
        assert TS.GetFactor(T2).T == DF2.T
        assert TS.GetFactor(T2).p == DF2.p
        
        
    def test_get_missing_discount_factor(self):
         
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85)
        
        l = [DF2, DF1] 
        
        TS = TermStructure(l)
        
        T = t + timedelta(days=365*2)
        
        DF = TS.GetFactor(T) 
        r = DF.GetRate()
        
        r1 = DF1.GetRate() 
        r2 = DF2.GetRate() 
        
        assert r1 <= r <= r2 
        
    def test_get_missing_discount_factor2(self):
         
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85)
        
        l = [DF2, DF1] 
        
        TS = TermStructure(l)
        
        T = t + timedelta(days=180)
        
        DF = TS.GetFactor(T) 
        
        r = DF.GetRate()
        
        r1 = DF1.GetRate() 
        
        for item in TS.discount_factors:
            
            print(item)
            print()
        
        
        assert r <= r1
    
    def test_get_forward(self):
        
        t = datetime.now() 
        
        T1 = t + timedelta(days=365)
        T2 = t + timedelta(days=365*3)
        
        DF1 = DiscountFactor(t, T1, .95)
        DF2 = DiscountFactor(t, T2, .85)
        
        l = [DF2, DF1] 
        
        TS = TermStructure(l)
        
        DF12 = ComputeForwardDiscount(DF1, DF2)  
        
        DF12_from_TS = TS.GetForward(T1, T2)
        
        assert (DF12.t == DF12_from_TS.t)
        assert (DF12.T == DF12_from_TS.T)   
        np.testing.assert_allclose(DF12.p, DF12_from_TS.p, rtol=1e-2, atol=1e-2) 
        
