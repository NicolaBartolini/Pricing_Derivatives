# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:57:27 2026

@author: Nicola
"""

import numpy as np 
from datetime import datetime, timedelta 

from contracts.options.EuropeanOption_class import EuroCall 
from contracts.options.ImpVolJackel import black_implied_volatility
from process.BlackScholesFormulas import EuroCall_BS, Black76Call, EuroPut_BS, Black76Put


import pytest 


class TestImpVolJackel:
    
    def test_otm_call_zero_rate(self):
        # zero rate case 
        
        F = 1 
        K = 1.1 
        sigma = .2 
        T = 1
        
        option_price = EuroCall_BS(F, K, T, 0, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T)
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_otm_call_non_zero_rate(self):
        
        S0 = 1
        K = 1.1
        sigma = .2 
        T = 1
        r = .025
        
        F = S0 * np.exp(r*T)
        DF = np.exp(-r*T)
        
        option_price = Black76Call(F, K, DF, T, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T)
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_otm_call_non_zero_rate_and_dividends(self):
        
        S0 = 1
        K = 1.1
        sigma = .2 
        T = 1
        r = .025
        q = .035
        
        F = S0 * np.exp((r-q)*T)
        DF = np.exp(-r*T)
        
        option_price = Black76Call(F, K, DF, T, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T)
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2) 
    
    def test_itm_call_zero_rate(self):
        # zero rate case 
        
        F = 1.2
        K = 1.1 
        sigma = .2 
        T = 1
        
        option_price = EuroCall_BS(F, K, T, 0, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T)
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_itm_call_non_zero_rate(self):
        
        S0 = 1.2
        K = 1.1
        sigma = .2 
        T = 1
        r = .025
        
        F = S0 * np.exp(r*T)
        DF = np.exp(-r*T)
        
        option_price = Black76Call(F, K, DF, T, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T)
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_itm_call_non_zero_rate_and_dividends(self):
        
        S0 = 1.2
        K = 1.1
        sigma = .2 
        T = 1
        r = .025
        q = .035
        
        F = S0 * np.exp((r-q)*T)
        DF = np.exp(-r*T)
        
        option_price = Black76Call(F, K, DF, T, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T)
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2) 
    
    
    ### test put 
    
    def test_itm_put_zero_rate(self):
        # zero rate case 
        
        F = 1 
        K = 1.1 
        sigma = .2 
        T = 1
        
        option_price = EuroPut_BS(F, K, T, 0, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T, flag='p')
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_itm_put_non_zero_rate(self):
        
        S0 = 1
        K = 1.1
        sigma = .2 
        T = 1
        r = .025
        
        F = S0 * np.exp(r*T)
        DF = np.exp(-r*T)
        
        option_price = Black76Put(F, K, DF, T, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T, flag='p')
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_itm_put_non_zero_rate_and_dividends(self):
        
        S0 = 1
        K = 1.1
        sigma = .2 
        T = 1
        r = .025
        q = .035
        
        F = S0 * np.exp((r-q)*T)
        DF = np.exp(-r*T)
        
        option_price = Black76Put(F, K, DF, T, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T, flag='p')
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_otm_put_zero_rate(self):
        # zero rate case 
        
        F = 1.2
        K = 1.1 
        sigma = .2 
        T = 1
        
        option_price = EuroPut_BS(F, K, T, 0, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T, flag='p')
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_otm_put_non_zero_rate(self):
        
        S0 = 1.2
        K = 1.1
        sigma = .2 
        T = 1
        r = .025
        
        F = S0 * np.exp(r*T)
        DF = np.exp(-r*T)
        
        option_price = Black76Put(F, K, DF, T, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T, flag='p')
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2)
    
    def test_otm_put_non_zero_rate_and_dividends(self):
        
        S0 = 1.2
        K = 1.1
        sigma = .2 
        T = 1
        r = .025
        q = .035
        
        F = S0 * np.exp((r-q)*T)
        DF = np.exp(-r*T)
        
        option_price = Black76Put(F, K, DF, T, sigma)
        
        vol = black_implied_volatility(option_price, F, K, T, flag='p')
        
        np.testing.assert_allclose(vol, sigma, atol=1e-2, rtol=1e-2) 
        
        
        
        