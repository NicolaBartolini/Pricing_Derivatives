# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:53:19 2026

@author: Nicola
"""

import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join("\\".join(root)))

import numpy as np 
from EuropeanOption_class import EuroCall, EuroPut
from PricingEngine_class import *
from process_class import GBM
from datetime import datetime, timedelta

import pytest 


@pytest.mark.parametrize("sigma", np.linspace(0.10, 0.50, 3))
@pytest.mark.parametrize("T", [0.25, 0.50, 1.00])
@pytest.mark.parametrize("K", [0.9, 1.0, 1.1])
def test_pricer_class_EuroCall(sigma, T, K):

    S0 = 1.0
    r = 0.05
    
    n_days = 365
    n = 18
    
    process = GBM(r, sigma)
    
    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=int(n_days * T))
    
    option = EuroCall(K, maturity)
    
    pricer1 = MonteCarloPricer(n_days, n) # monte-carlo pricer 
    pricer2 = FourierDampingPricer(1.5)
    # pricer3 = CosPricer()
    
    X0 = [S0]
    
    mc_price = pricer1.evaluate_option(option, process, X0, r, today)
    price2 = pricer2.evaluate_option(option, process, X0, r, today)
    
    print(f"MC      price = {mc_price}")
    print(f"Other   price = {price2}")
    print(f"Difference = {mc_price - price2}")
    print()

    
    np.testing.assert_allclose(mc_price, price2, rtol=1e-3, atol=1e-3)
  

@pytest.mark.parametrize("sigma", np.linspace(0.10, 0.50, 3))
@pytest.mark.parametrize("T", [0.25, 0.50, 1.00])
@pytest.mark.parametrize("K", [0.9, 1.0, 1.1])
def test_pricer_class_EuroCall2(sigma, T, K):

    S0 = 1.0
    r = 0.05
    
    n_days = 365
    n = 18
    
    process = GBM(r, sigma)
    
    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=int(n_days * T))
    
    option = EuroCall(K, maturity)
    
    pricer1 = MonteCarloPricer(n_days, n) # monte-carlo pricer 
    # pricer2 = FourierDampingPricer(1.5)
    pricer2 = CosPricer()
    
    X0 = [S0]
    
    mc_price = pricer1.evaluate_option(option, process, X0, r, today)
    price2 = pricer2.evaluate_option(option, process, X0, r, today)
    
    print(f"MC      price = {mc_price}")
    print(f"other   price = {price2}")
    print(f"Difference = {mc_price - price2}")
    print()

    np.testing.assert_allclose(mc_price, price2, rtol=1e-3, atol=1e-3)
    


@pytest.mark.parametrize("sigma", np.linspace(0.10, 0.50, 3))
@pytest.mark.parametrize("T", [0.25, 0.50, 1.00])
@pytest.mark.parametrize("K", [0.9, 1.0, 1.1])
def test_pricer_class_EuroPut(sigma, T, K):

    S0 = 1.0
    r = 0.05
    
    n_days = 365
    n = 18
    
    process = GBM(r, sigma)
    
    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=int(n_days * T))
    
    option = EuroPut(K, maturity)
    
    pricer1 = MonteCarloPricer(n_days, n) # monte-carlo pricer 
    pricer2 = FourierDampingPricer(-1.5)
    # pricer3 = CosPricer()
    
    X0 = [S0]
    
    mc_price = pricer1.evaluate_option(option, process, X0, r, today)
    price2 = pricer2.evaluate_option(option, process, X0, r, today)
    
    print(f"MC      price = {mc_price}")
    print(f"Other   price = {price2}")
    print(f"Difference = {mc_price - price2}")
    print()

    
    np.testing.assert_allclose(mc_price, price2, rtol=1e-3, atol=1e-3)
    

@pytest.mark.parametrize("sigma", np.linspace(0.10, 0.50, 3))
@pytest.mark.parametrize("T", [0.25, 0.50, 1.00])
@pytest.mark.parametrize("K", [0.9, 1.0, 1.1])
def test_pricer_class_EuroPut2(sigma, T, K):

    S0 = 1.0
    r = 0.05
    
    n_days = 365
    n = 18
    
    process = GBM(r, sigma)
    
    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=int(n_days * T))
    
    option = EuroPut(K, maturity)
    
    pricer1 = MonteCarloPricer(n_days, n) # monte-carlo pricer 
    # pricer2 = FourierDampingPricer(1.5)
    pricer2 = CosPricer()
    
    X0 = [S0]
    
    mc_price = pricer1.evaluate_option(option, process, X0, r, today)
    price2 = pricer2.evaluate_option(option, process, X0, r, today)
    
    print(f"MC      price = {mc_price}")
    print(f"other   price = {price2}")
    print(f"Difference = {mc_price - price2}")
    print()

    np.testing.assert_allclose(mc_price, price2, rtol=1e-3, atol=1e-3)
