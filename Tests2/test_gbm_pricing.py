# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:24:50 2026

@author: Nicola
"""

import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))

import numpy as np
import pytest

from datetime import datetime, timedelta
from scipy.stats import norm

from process_class import GBM
from EuropeanOption_class import EuroCall, EuroPut, EuroCallDigital, EuroPutDigital
from EuropeanOption_class import EuroCallAssetOrNothing
from PricingEngine_class import CosPricer, MonteCarloPricer, FourierDampingPricer
from BlackScholesFormulas import *



class TestGBMPricing:
    
    @pytest.fixture
    def market_params(self):
        """Fixture providing shared baseline market and option parameters."""
        D =  {"S0": 1.0, "K": 1.0, "r": 0.05, "today": datetime(2026, 1, 1),
              "N": 256, "L": 10, 'nPath' : 15, 'nSteps' : 252, 'alpha' : 1.5}
        
        cos_pricer = CosPricer(D['N'], D['L'])
        mc_pricer = MonteCarloPricer(D['nSteps'], D['nPath'])
        damping_pricer = FourierDampingPricer(D['alpha'])
        
        D['cos_pricer'] = cos_pricer
        D['mc_pricer'] = mc_pricer
        D['damping_pricer'] = damping_pricer
        
        return D
    
    @pytest.mark.parametrize("sigma", [0.20, 0.40])
    @pytest.mark.parametrize("T", [.25, 0.50, 1.00])
    def test_cos_matches_black_scholes_eurocall(self, market_params, sigma, T):
        """Test that the COS call price matches the Black-Scholes analytical formula."""
        p = market_params
        
        maturity = p["today"] + timedelta(days=int(365 * T))

        contract = EuroCall(p["K"], maturity)
        model = GBM(p["r"], sigma)
        
        cos_price = p['cos_pricer'].evaluate_option(contract, model, X0=[p["S0"]], r=p["r"], day=p['today'] )
        
        bs_call = EuroCall_BS(p["S0"], p["K"], T, p["r"], sigma)

        np.testing.assert_allclose(cos_price, bs_call, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.parametrize("sigma", [0.20, 0.40])
    @pytest.mark.parametrize("T", [.25, 0.50, 1.00])
    def test_cos_matches_black_scholes_europut(self, market_params, sigma, T):
        """Test that the COS put price matches the Black-Scholes analytical formula."""

        p = market_params
        
        maturity = p["today"] + timedelta(days=int(365 * T))

        contract = EuroPut(p["K"], maturity)
        model = GBM(p["r"], sigma)
        
        cos_price = p['cos_pricer'].evaluate_option(contract, model, X0=[p["S0"]], r=p["r"], day=p['today'] )
        
        bs_price = EuroPut_BS(p["S0"], p["K"], T, p["r"], sigma)

        np.testing.assert_allclose(cos_price, bs_price, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.parametrize("sigma", [0.20, 0.40])
    @pytest.mark.parametrize("T", [.25, 0.50, 1.00])
    def test_cos_matches_black_scholes_eurocall_digital(self, market_params, sigma, T):
        """Test that the COS eurocall digital price matches the Black-Scholes analytical formula."""

        p = market_params
        
        maturity = p["today"] + timedelta(days=int(365 * T))

        contract = EuroCallDigital(p["K"], maturity)
        model = GBM(p["r"], sigma)
        
        cos_price = p['cos_pricer'].evaluate_option(contract, model, X0=[p["S0"]], r=p["r"], day=p['today'] )
        
        bs_price = EuroCallDigital_BS(p["S0"], p["K"], T, p["r"], sigma)

        np.testing.assert_allclose(cos_price, bs_price, rtol=1e-2, atol=1e-2)
        
    
    @pytest.mark.parametrize("sigma", [0.20, 0.40])
    @pytest.mark.parametrize("T", [.25, 0.50, 1.00])
    def test_cos_matches_black_scholes_europut_digital(self, market_params, sigma, T):
        """Test that the COS europut digital price matches the Black-Scholes analytical formula."""

        p = market_params
        
        maturity = p["today"] + timedelta(days=int(365 * T))

        contract = EuroPutDigital(p["K"], maturity)
        model = GBM(p["r"], sigma)
        
        cos_price = p['cos_pricer'].evaluate_option(contract, model, X0=[p["S0"]], r=p["r"], day=p['today'] )
        
        bs_price = EuroPutDigital_BS(p["S0"], p["K"], T, p["r"], sigma)

        np.testing.assert_allclose(cos_price, bs_price, rtol=1e-2, atol=1e-2)
        
    
    # @pytest.mark.parametrize("sigma", [0.20, 0.40])
    # @pytest.mark.parametrize("T", [.25, 0.50, 1.00])
    # def test_cos_matches_black_scholes_eurocall_AON(self, market_params, sigma, T):
    #     """Test that the COS put price matches the Black-Scholes analytical formula."""

    #     p = market_params
        
    #     maturity = p["today"] + timedelta(days=int(365 * T))

    #     contract = EuroCallAssetOrNothing(p["K"], maturity)
    #     model = GBM(p["r"], sigma)
        
    #     cos_price = p['cos_pricer'].evaluate_option(contract, model, X0=[p["S0"]], r=p["r"], day=p['today'] )
        
    #     bs_price = EuroCallAssetOrNothing_BS(p["S0"], p["K"], T, p["r"], sigma)

    #     np.testing.assert_allclose(cos_price, bs_price, rtol=1e-2, atol=1e-2) 
    
    #### Damping method now 
    
    @pytest.mark.parametrize("sigma", [0.20, 0.40])
    @pytest.mark.parametrize("T", [.25, 0.50, 1.00])
    def test_damping_matches_black_scholes_eurocall(self, market_params, sigma, T):
        """Test that the COS call price matches the Black-Scholes analytical formula."""
        p = market_params
        
        maturity = p["today"] + timedelta(days=int(365 * T))

        contract = EuroCall(p["K"], maturity)
        model = GBM(p["r"], sigma)
        
        pricer = FourierDampingPricer(p['alpha'])
        
        fourier_price = pricer.evaluate_option(contract, model, X0=[p["S0"]], r=p["r"], day=p['today'] )
        
        bs_call = EuroCall_BS(p["S0"], p["K"], T, p["r"], sigma)

        np.testing.assert_allclose(fourier_price, bs_call, rtol=1e-2, atol=1e-2)
    
    @pytest.mark.parametrize("sigma", [0.20, 0.40])
    @pytest.mark.parametrize("T", [.25, 0.50, 1.00])
    def test_damping_matches_black_scholes_europut(self, market_params, sigma, T):
        """Test that the COS put price matches the Black-Scholes analytical formula."""

        p = market_params
        
        maturity = p["today"] + timedelta(days=int(365 * T))

        contract = EuroPut(p["K"], maturity)
        model = GBM(p["r"], sigma)
        
        pricer = FourierDampingPricer(-p['alpha'])
        
        fourier_price = pricer.evaluate_option(contract, model, X0=[p["S0"]], r=p["r"], day=p['today'] )
        
        bs_price = EuroPut_BS(p["S0"], p["K"], T, p["r"], sigma)

        np.testing.assert_allclose(fourier_price, bs_price, rtol=1e-2, atol=1e-2)
        
        