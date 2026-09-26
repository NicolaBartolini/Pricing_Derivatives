# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:37:02 2026

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

from process_class import Bates
from EuropeanOption_class import EuroCall, EuroPut, EuroCallDigital, EuroPutDigital, EuroCallAssetOrNothing, EuroPutAssetOrNothing
from EuropeanOption_class import EuroCallAssetOrNothing
from PricingEngine_class import CosPricer, MonteCarloPricer, FourierDampingPricer, MonteCarloPricer
from BlackScholesFormulas import *
from AsianOption_class import AsianCall , AsianPut


np.random.seed(42)

class TestBatesPricing:
    
    @pytest.fixture
    def inputs(self):
        """Fixture providing shared baseline market and option parameters.""" 
        
        # v0, rho, eta, theta, kappa
        # (0.04, -0.30, 0.20, 0.04, 1.0)
        
        # kappa, theta, eta, rho, lambda, muJ, sigmaJ
        # 2.0, 0.04, 0.2, -0.30, 0.20, -0.05, 0.20
        
        r = .05
        
        model = Bates(mu=r, kappa=2., theta=.04, eta=.2, rho=-.3, lam=.2, muj=-.05, sigmaj=.2)
        
        D =  {"S0": 1.0, "K": 1.0, "r": r, 'v0' : 0.04, 'model' : model,
              "today": datetime(2026, 1, 1),
              "N": 256, "L": 10, 'nPath' : 18, 'nSteps' : 252, 'alpha' : 1.5}
        
        cos_pricer = CosPricer(D['N'], D['L'])
        mc_pricer = MonteCarloPricer(D['nSteps'], D['nPath'])
        # damping_pricer = FourierDampingPricer(D['alpha'])
        
        D['cos_pricer'] = cos_pricer
        D['mc_pricer'] = mc_pricer
        # D['damping_pricer'] = damping_pricer
        
        return D 
    
    STRIKES = [.8, 1, 1.2]
    @pytest.mark.parametrize("K", STRIKES)
    def test_bates_mc_vs_cos_eurocall(self,inputs, K):

        
        maturity = inputs['today'] + timedelta(days=365)

        contract = EuroCall(K, maturity)

        model = inputs['model']
        X0 = [inputs['S0'], inputs['v0']]
        
        # cos_price = CosPricingEngine(option=option, process=process, X0=[S0,v0], r=r, N=256, L=10, day=today)
        cos_price = inputs['cos_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

        # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
        mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )


        print(f"COS price : {cos_price}")
        print(f"MC  price : {mc_price}")
        print()
        
        np.testing.assert_allclose(mc_price, cos_price, rtol=1e-2, atol=1e-2)
        
    STRIKES = [.8, 1, 1.2]
    @pytest.mark.parametrize("K", STRIKES)
    def test_bates_mc_vs_cos_europut(self,inputs, K):

        
        maturity = inputs['today'] + timedelta(days=365)

        contract = EuroPut(K, maturity)

        model = inputs['model']
        X0 = [inputs['S0'], inputs['v0']]
        
        # cos_price = CosPricingEngine(option=option, process=process, X0=[S0,v0], r=r, N=256, L=10, day=today)
        cos_price = inputs['cos_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

        # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
        mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )


        print(f"COS price : {cos_price}")
        print(f"MC  price : {mc_price}")
        print()
        
        np.testing.assert_allclose(mc_price, cos_price, rtol=1e-2, atol=1e-2)
    
    STRIKES = [.8, 1, 1.2]
    @pytest.mark.parametrize("K", STRIKES)
    def test_bates_mc_vs_damping_eurocall(self,inputs, K):

        
        maturity = inputs['today'] + timedelta(days=365)

        contract = EuroCall(K, maturity)

        model = inputs['model']
        X0 = [inputs['S0'], inputs['v0']]
        
        damping_pricer = FourierDampingPricer(inputs['alpha'])

        damping_price = damping_pricer.evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

        # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
        mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )


        print(f"COS price : {damping_price}")
        print(f"MC  price : {mc_price}")
        print()
        
        np.testing.assert_allclose(mc_price, damping_price, rtol=1e-2, atol=1e-2)
        
    STRIKES = [.8, 1, 1.2]
    @pytest.mark.parametrize("K", STRIKES)
    def test_bates_mc_vs_damping_europut(self,inputs, K):

        
        maturity = inputs['today'] + timedelta(days=365)

        contract = EuroPut(K, maturity)

        model = inputs['model']
        X0 = [inputs['S0'], inputs['v0']]
        
        damping_pricer = FourierDampingPricer(-inputs['alpha'])

        damping_price = damping_pricer.evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

        # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
        mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )


        print(f"COS price : {damping_price}")
        print(f"MC  price : {mc_price}")
        print()
        
        np.testing.assert_allclose(mc_price, damping_price, rtol=1e-2, atol=1e-2)
    
    
    STRIKES = [.8, 1, 1.2]
    @pytest.mark.parametrize("K", STRIKES)
    def test_bates_mc_vs_cos_calldigital(self,inputs, K):

        
        maturity = inputs['today'] + timedelta(days=365)

        contract = EuroCallDigital(K, maturity)

        model = inputs['model']
        X0 = [inputs['S0'], inputs['v0']]
        
        # cos_price = CosPricingEngine(option=option, process=process, X0=[S0,v0], r=r, N=256, L=10, day=today)
        cos_price = inputs['cos_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

        # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
        mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )
        
        print(f"COS price : {cos_price}")
        print(f"MC  price : {mc_price}")
        print()
        
        np.testing.assert_allclose(mc_price, cos_price, rtol=1e-2, atol=1e-2)
    
    STRIKES = [.8, 1, 1.2]
    @pytest.mark.parametrize("K", STRIKES)
    def test_bates_mc_vs_cos_putdigital(self,inputs, K):

        
        maturity = inputs['today'] + timedelta(days=365)

        contract = EuroPutDigital(K, maturity)

        model = inputs['model']
        X0 = [inputs['S0'], inputs['v0']]
        
        # cos_price = CosPricingEngine(option=option, process=process, X0=[S0,v0], r=r, N=256, L=10, day=today)
        cos_price = inputs['cos_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

        # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
        mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )


        print(f"COS price : {cos_price}")
        print(f"MC  price : {mc_price}")
        print()
        
        np.testing.assert_allclose(mc_price, cos_price, rtol=1e-2, atol=1e-2)
    
    
    STRIKES = [.8, 1, 1.2]
    @pytest.mark.parametrize("K", STRIKES)
    def test_bates_mc_vs_damping_euroAON(self,inputs, K):

        
        maturity = inputs['today'] + timedelta(days=365)

        contract = EuroCallAssetOrNothing(K, maturity)

        model = inputs['model']
        X0 = [inputs['S0'], inputs['v0']]
        
        damping_pricer = FourierDampingPricer(inputs['alpha'])

        damping_price = damping_pricer.evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

        # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
        mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )


        print(f"COS price : {damping_price}")
        print(f"MC  price : {mc_price}")
        print()
        
        np.testing.assert_allclose(mc_price, damping_price, rtol=1e-2, atol=1e-2)
        
    STRIKES = [.8, 1, 1.2]
    @pytest.mark.parametrize("K", STRIKES)
    def test_bates_mc_vs_damping_europutAON(self,inputs, K):

        
        maturity = inputs['today'] + timedelta(days=365)

        contract = EuroPutAssetOrNothing(K, maturity)

        model = inputs['model']
        X0 = [inputs['S0'], inputs['v0']]
        
        damping_pricer = FourierDampingPricer(-inputs['alpha'])

        damping_price = damping_pricer.evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

        # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
        mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )


        print(f"COS price : {damping_price}")
        print(f"MC  price : {mc_price}")
        print()
        
        np.testing.assert_allclose(mc_price, damping_price, rtol=1e-2, atol=1e-2)
   

# def fun():
    
#     # v0, rho, eta, theta, kappa
#     (0.04, -0.30, 0.20, 0.04, 1.0)
    
#     r = .05
    
#     model = Heston93(mu=r, kappa=.1, theta=.04,eta=.2, rho=-.3)
    
#     inputs =  {"S0": 1.0, "K": 1.0, "r": r, 'v0' : 0.04, 'model' : model,
#           "today": datetime(2026, 1, 1),
#           "N": 256, "L": 10, 'nPath' : 16, 'nSteps' : 252, 'alpha' : 1.5}
    
#     cos_pricer = CosPricer(inputs['N'], inputs['L'])
#     mc_pricer = MonteCarloPricer(inputs['nSteps'], inputs['nPath'])
#     # damping_pricer = FourierDampingPricer(D['alpha'])
    
#     inputs['cos_pricer'] = cos_pricer
#     inputs['mc_pricer'] = mc_pricer
#     # D['damping_pricer'] = damping_pricer 
    
      
#     maturity = inputs['today'] + timedelta(days=365)

#     contract = EuroCall(1., maturity)

#     model = inputs['model']
#     X0 = [inputs['S0'], inputs['v0']]
    
#     # cos_price = CosPricingEngine(option=option, process=process, X0=[S0,v0], r=r, N=256, L=10, day=today)
#     cos_price = inputs['cos_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )

#     # mc_price = MonteCarloEngine(option=option, process=process, X0=[S0,v0], r=r, n_steps=252, n=16, day=today)
#     mc_price = inputs['mc_pricer'].evaluate_option(contract, model, X0=X0, r=inputs["r"], day=inputs['today'] )


#     print(f"COS price : {cos_price}")
#     print(f"MC  price : {mc_price}")
#     print()
    
#     np.testing.assert_allclose(mc_price, cos_price, rtol=1e-2, atol=1e-2)
    

# fun()