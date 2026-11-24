# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:57:55 2026

@author: Nicola
"""

import numpy as np
import pytest

from datetime import datetime, timedelta
from scipy.stats import norm

from process.process_class import Black76 

from options.EuropeanOption_class import Capl, Floorl, Cap, Floor
from pricing.PricingEngine_class import MonteCarloPricer

from process.BlackScholesFormulas import *


np.random.seed(42)


class TestLiborPricing:
    
    @pytest.fixture
    def market_params(self):
        """Fixture providing shared baseline market and option parameters."""
        D =  {"L0": .025, "K": .03, "r": 0.02, "today": datetime(2026, 1, 1),
              "N": 256, "L": 10, 'nPath' : 15, 'nSteps' : 252}
        
        mc_pricer = MonteCarloPricer(D['nSteps'], D['nPath'])
        
        D['mc_pricer'] = mc_pricer
        # D['damping_pricer'] = damping_pricer
        
        return D
    
    @pytest.mark.parametrize("sigma", [0.05, .1, 0.2])
    @pytest.mark.parametrize("T", [.25, 0.50, 1.00]) 
    def test_mc_matches_exact(self, market_params, sigma, T):
        
        p = market_params 
        alpha = .25  # alpha=.25 quarterly Libor
        
        maturity = p["today"] + timedelta(days=int(365 * T))

        contract = Capl(p["K"], maturity, alpha=alpha)
        model = Black76(sigma)
        
        # get the discount factor 
        DF = np.exp(-T * p['r']) # In the libor market models we price under Q^T however here for semplicity we recover P(t,T) from the formula e^{-r*T}
                                 # but in reality we should recover r corresponding to P(t,T) in this way I can reuse the former pricing engine
                                 
        ###• Monte-carlo price
        
        pricer = p['mc_pricer']
        
        mc_price = pricer.evaluate_option(contract, model, [p['L0']], 0, p["today"])
        
        exact = EuroCall_BS(p["L0"], p["K"], T, 0, sigma) * DF * alpha  # here we consider a unitary notional 
        
        np.testing.assert_allclose(mc_price, exact, rtol=1e-2, atol=1e-2)
        
    @pytest.mark.parametrize("sigma", [0.05, .1, 0.2])
    @pytest.mark.parametrize("T", [.25, 0.50, 1.00]) 
    def test_mc_floorl_matches_exact(self, market_params, sigma, T):
        
        p = market_params 
        alpha = .25  # alpha=.25 quarterly Libor
        
        maturity = p["today"] + timedelta(days=int(365 * T))

        contract = Floorl(p["K"], maturity, alpha=alpha)
        model = Black76(sigma)
        
        # get the discount factor 
        DF = np.exp(-T * p['r']) # In the libor market models we price under Q^T however here for semplicity we recover P(t,T) from the formula e^{-r*T}
                                 # but in reality we should recover r corresponding to P(t,T) in this way I can reuse the former pricing engine
                                 
        ### Monte-carlo price
        
        pricer = p['mc_pricer']
        
        mc_price = pricer.evaluate_option(contract, model, [p['L0']], 0, p["today"])
        
        exact = EuroPut_BS(p["L0"], p["K"], T, 0, sigma) * DF * alpha  # here we consider a unitary notional 
        
        np.testing.assert_allclose(mc_price, exact, rtol=1e-2, atol=1e-2) 
    
    
    def test_cap(self, market_params):
                
        T = [.25, 0.50, 1.00]
        DF = [np.exp(-.03 * T[0]), np.exp(-.03 * T[1]), np.exp(-.03 * T[2])]
        alpha = .25
        
        list_of_capl = []
        
        for i in range(0, 3):
            
            maturity = market_params["today"] + timedelta(days=int(365 * T[i]))
            list_of_capl.append(Capl(market_params["K"], maturity, alpha=alpha))
        
        contract = Cap(list_of_capl, DF) 
        
        assert isinstance(contract, Cap)
    
    def test_floor(self, market_params):
                
        T = [.25, 0.50, 1.00]
        DF = [np.exp(-.03 * T[0]), np.exp(-.03 * T[1]), np.exp(-.03 * T[2])]
        alpha = .25
        
        list_of_capl = []
        
        for i in range(0, 3):
            
            maturity = market_params["today"] + timedelta(days=int(365 * T[i]))
            list_of_capl.append(Capl(market_params["K"], maturity, alpha=alpha))
        
        contract = Floor(list_of_capl, DF) 
        
        assert isinstance(contract, Floor)
        
        
        
        
    
   
   
