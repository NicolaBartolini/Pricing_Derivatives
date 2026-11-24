# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 08:29:14 2026

@author: Nicola
"""


import numpy as np
import pytest 

from process.additive_process import CIR 
from contracts.bonds.bonds import ZCB, CB
from datetime import datetime, timedelta
from pricing.BondPricerEngine import PriceZCB, PriceCB

np.random.seed(42)

class TestVasicekBondPricing:
    
    @pytest.fixture
    def model_instance(self):
        """Fixture providing shared baseline market and option parameters."""
        # params =  {'x0' : .01, 'kappa' : 1.5, 'theta' : .025, 'sigma' : .2}
        
        model = CIR(kappa=1.5, theta=.025, sigma=.2)
    
        return model 
    
    def test_zcb_price1(self, model_instance):
        
        price0 = np.inf
        day = datetime.now() 
        r0 = .025 
        
        # print()
        
        for T in range(1, 10):
            
            maturity_date = datetime.now() + timedelta(T*365)
            
            zcb_contract = ZCB(maturity_date)
            
            price1 = PriceZCB(zcb_contract, model_instance, r0, day)
            
            assert(price1<=price0)
            
            # price0 = price1 
            # print(price0) 
        
    
    def test_zcb_price2(self, model_instance):
        
        day = datetime.now() 
        r0 = .025 
            
        maturity_date = datetime.now() 
        
        zcb_contract = ZCB(maturity_date)
        
        price = PriceZCB(zcb_contract, model_instance, r0, day)
        
        np.testing.assert_allclose(price, 1, rtol=1e-2, atol=1e-2)
    
    def test_zcb_price3(self, model_instance):
        
        day = datetime.now() 
        r0 = .025 
            
        maturity_date = datetime.now() + timedelta(1000*365)
        
        zcb_contract = ZCB(maturity_date)
        
        price = PriceZCB(zcb_contract, model_instance, r0, day)
        
        np.testing.assert_allclose(price, 0, atol=1e-2)
        
        
        
            