# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:49:53 2026

@author: Nicola
"""

import numpy as np 
from datetime import datetime, timedelta 

from contracts.options.EuropeanOption_class import Capl, Floorl 
from contracts.options.ImpliedVolFormulas import imp_atm_sigma_hagan2002, imp_vol_sigma_hagan2002
from process.BlackScholesFormulas import Black76Capl, Black76Floorl 
from process.process_class import Sabr 
from pricing.PricingEngine_class import MonteCarloPricer 
from numerical_methods.numerical_methods import bisection_method

import pytest 

np.random.seed(42)


class TestsImpVol:
    
    @pytest.mark.parametrize('T', [.25, .5, 1])
    def test_atm_imp_vol_capl_hagan2002(self, T):
        
        # T = .25
        X0 = [.025, .1] # initial value [F0, alpha0] to be consistent with Hagan 2002 notation 
        K = .025
        N = 18 # 2^N monte-carlo simulations 
        n_days = int(252 * T) # number of steps
        
        # initialize the Black76 model 
        model = Sabr(beta=.5, sigma=.1, rho=-.1)
        
        date = datetime(2026, 1, 5)
        maturity_date = date + timedelta(days=n_days+1)
        
        option = Capl(K, maturity_date, 1)
        
        # let's express the discount factor in terms of the rate r (to use the monte-carlo pricer)
        r = .01
        
        T_actual = option.get_tenor(date)

        DF = np.exp(-r * T_actual)
        
        ####
        
        n_steps = int(T*252)
        pricer = MonteCarloPricer(n_steps, N)
        
        ### let's compute the price of the option with the sabr
        
        option_price = pricer.evaluate_option(option, model, X0, r, date) * 100
        
        ### let's compute the implied vol via bisection formula
        
        fun = lambda sigma : option_price - Black76Capl(X0[0], K, DF, T_actual, sigma, 1, 100)
        
        imp_vol_bis   = bisection_method(fun, .01, 2.5) 
        imp_vol_hagan = imp_atm_sigma_hagan2002(X0[1], X0[0], .1, .5, -.1, T)
        
        print()
        print(f'Actual  : {imp_vol_bis}')
        print(f'Desired : {imp_vol_hagan}')
        print()
        
        # np.testing.assert_allclose(imp_vol_hagan, imp_vol_bis, atol=1, rtol=1)
        np.testing.assert_allclose(imp_vol_hagan, imp_vol_bis, atol=1e-3, rtol=1e-3)
    
    @pytest.mark.parametrize('T', [.25, .5, 1])
    def test_imp_vol_capl_hagan2002(self, T):
        
        # T = .25
        X0 = [.025, .1] # initial value [F0, alpha0] to be consistent with Hagan 2002 notation 
        K = .022
        N = 18 # 2^N monte-carlo simulations 
        n_days = int(252 * T) # number of steps
        
        # initialize the Black76 model 
        model = Sabr(beta=.5, sigma=.1, rho=-.1)
        
        date = datetime(2026, 1, 5)
        maturity_date = date + timedelta(days=n_days+1)
        
        option = Capl(K, maturity_date, 1)
        
        # let's express the discount factor in terms of the rate r (to use the monte-carlo pricer)
        r = .01
        
        T_actual = option.get_tenor(date)

        DF = np.exp(-r * T_actual)
        
        ####
        
        n_steps = int(T*252)
        pricer = MonteCarloPricer(n_steps, N)
        
        ### let's compute the price of the option with the sabr
        
        option_price = pricer.evaluate_option(option, model, X0, r, date) * 100
        
        ### let's compute the implied vol via bisection formula
        
        fun = lambda sigma : option_price - Black76Capl(X0[0], K, DF, T_actual, sigma, 1, 100)
        
        imp_vol_bis   = bisection_method(fun, .01, 2.5) 
        imp_vol_hagan = imp_vol_sigma_hagan2002(X0[1], X0[0], K, .1, .5, -.1, T)
        
        print()
        print(f'Actual  : {imp_vol_bis}')
        print(f'Desired : {imp_vol_hagan}')
        print()
        
        # np.testing.assert_allclose(imp_vol_hagan, imp_vol_bis, atol=1, rtol=1)
        np.testing.assert_allclose(imp_vol_hagan, imp_vol_bis, atol=1e-3, rtol=1e-3)
                
        
        
        
        
        