# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:42:43 2026

@author: Nicola
"""

import numpy as np
import pytest 

from process.additive_process import OrnsteinUhlenbeck 
from contracts.bonds.bonds import ZCB, CB
from datetime import datetime, timedelta
from pricing.BondPricerEngine import PriceZCB, PriceCB

np.random.seed(42)

class TestVasicekBondPricing:
    
    @pytest.fixture
    def model_instance(self):
        """Fixture providing shared baseline market and option parameters."""
        # params =  {'x0' : .01, 'kappa' : 1.5, 'theta' : .025, 'sigma' : .2}
        
        model = OrnsteinUhlenbeck(kappa=1.5, theta=.025, sigma=.2)
    
        return model 
    
    @pytest.mark.parametrize('T', range(1, 10, 1))
    def test_zcb_price(self, model_instance, T):
        
        maturity_date = datetime.now() + timedelta(T*365)
        day = datetime.now() 
        
        zcb_contract = ZCB(maturity_date)
        
        tenor = zcb_contract.get_tenor(day)
        
        r0 = .025 
        
        price = PriceZCB(zcb_contract, model_instance, r0, day)
        
        # Let's compare them with the Monte-Carlo price 
        kappa = model_instance.kappa 
        theta = model_instance.theta 
        sigma = model_instance.sigma 
        
        M = theta * tenor + (r0 - theta) *  (1 - np.exp(-kappa*tenor))/kappa
        S = (sigma/kappa)**2 * (tenor - 2 * (1 - np.exp(-kappa*tenor))/kappa + (1 - np.exp(-2*kappa*tenor))/(2*kappa))
        
        trj = np.random.normal(M, np.sqrt(S), 5000)
        MC_price = np.mean(np.exp(-trj))
        
        ### 
        
        np.testing.assert_allclose(price, MC_price, rtol = 1e-2, atol=1e-2) 
    
    def test_cb_price(self, model_instance):
        
        T = 10
        N = 1<<13
        
        maturity_date = datetime.now() + timedelta(T*365)
        day = datetime.now() 
        coupon_rate = .03 
        
        # create the coupon payment schedule list to initialize the object insance CB
        payments_schedule = [day + timedelta(100)] 
        
        while  payments_schedule[-1] + timedelta(100) <= maturity_date:
            
            payments_schedule.append(payments_schedule[-1] + timedelta(100))
        
        print(payments_schedule[-1])
        print()
        print(maturity_date)
        
        cb_contract = CB(coupon_rate, payments_schedule, maturity_date)
        
        tenor = cb_contract.get_tenor(day)
        
        r0 = .025 
                
        price = PriceCB(cb_contract, model_instance, r0, day)
        
        # Let's compare them with the Monte-Carlo price 
        kappa = model_instance.kappa 
        theta = model_instance.theta 
        sigma = model_instance.sigma 
        
        M = theta * tenor + (r0 - theta) *  (1 - np.exp(-kappa*tenor))/kappa
        S = (sigma/kappa)**2 * (tenor - 2 * (1 - np.exp(-kappa*tenor))/kappa + (1 - np.exp(-2*kappa*tenor))/(2*kappa))
        
        # ZCB part of the price 
        
        trj = np.random.normal(M, np.sqrt(S), N)
        MC_price = np.mean(np.exp(-trj))
        
        for Ti in cb_contract.get_tenor_coupon_payments(day):
            
            Mi = theta * Ti + (r0 - theta) *  (1 - np.exp(-kappa*Ti))/kappa
            Si = (sigma/kappa)**2 * (Ti - 2 * (1 - np.exp(-kappa*Ti))/kappa + (1 - np.exp(-2*kappa*Ti))/(2*kappa))
            
            trj = np.random.normal(Mi, np.sqrt(Si), N)
            MC_price += coupon_rate * np.mean(np.exp(-trj))
            
        
        ### 
        
        np.testing.assert_allclose(price, MC_price, rtol = 1e-2, atol=1e-2) 
        
    
    
        
        
        
        
        