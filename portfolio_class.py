# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:32:17 2026

@author: Nicola
"""

from EuropeanOption_class import *
from AsianOption_class import * 
from AmericanOption_class import * 
from BarrierOption_class import * 
from PricingEngine_class import * 
from DeltaHedgingEngine import compute_Delta

import numpy as np 
from datetime import datetime



class PortfolioPosition:
    
    def __init__(self, option, process, X0, pricer, n_units, trading_day):
        
        self.option = option 
        self.process = process
        self.pricer = pricer
        self.X0 = X0 
        self.n_units = n_units
        self.trading_day = trading_day
    
    def evaluate(self, r):
        
        option_price = self.pricer.evaluate_option(self.option, self.process, self.X0, r, self.trading_day)
        
        return self.n_units * option_price 
    
    def compute_delta(self, r, dx=1e-4):
        
        delta = compute_Delta(self.option, self.process, self.pricer, self.X0, r, self.trading_day, dx)
        
        return self.n_units * delta
    
    def simulate(self, r, day, n):
        
        N = 1<<n # number of Monte-Carlo simulations
        n_steps = (day - self.trading_day).days
        t1 = n_steps/365
        
        # Simulate the process underying up to the VaR evaluation date
        sim = self.process.simulate( self.X0, t1, n_steps, n)
        
        # Store the option price at t1
        option_prices_at_t1 = np.empty(N)
        
        if isinstance(sim, tuple):
            St1 = sim[0][-1]      # Heston - Bates: (S, V)
            nut1 = sim[1][-1]
            
            for i in range(0, N):
                
                X1 = [St1[i], nut1[i]]
                option_prices_at_t1[i] = self.pricer.evaluate_option(self.option, self.process, X1, r, day)
            
        else:
            St1 = sim[-1]         # GBM - Merton - Variance GAmma: S 
            
            for i in range(0, N):
                
                X1 = [St1[i]]
                option_prices_at_t1[i] = self.pricer.evaluate_option(self.option, self.process, X1, r, day)
        
        return option_prices_at_t1 * self.n_units



class Portfolio:
    
    def __init__(self, assets : list[PortfolioPosition]):
        
        self.assets = assets 
        
    def evaluate(self, r : float):
        
        res = 0 
        
        for asset in self.assets:
            
            res += asset.evaluate(r) 
        
        return res 
    
    def VaR_ES(self, q, r, day, n):
        
        portfolio_value_at_t0 = self.evaluate(r)
        portfolio_value_at_t1 = np.zeros(1<<n)
        
        for asset in self.assets:
            
            portfolio_value_at_t1 += asset.simulate(r, day, n)
        
        Loss = portfolio_value_at_t0 - portfolio_value_at_t1
        
        VaR = np.quantile(Loss, q)
        ES = np.mean(Loss[Loss>= VaR])
        
        return VaR, ES, Loss


if __name__=='__main__':
    
    from datetime import timedelta
    from process_class import GBM
    
    r = .025
    trading_day = datetime(2026, 1, 4)
    
    ############## Position1
    
    sigma1 = .1 
    S01 = 1 
    K1 = 1
    pricer1 = CosPricer() 
    
    option1_maturity_date = trading_day + timedelta(365)
    
    option1 = EuroCall(K1, option1_maturity_date)
    process1 = GBM(r, sigma1)
    
    units1 = 1
    
    position1 = PortfolioPosition(option1, process1, [S01], pricer1, units1, trading_day)
    
    value_pos1 = position1.evaluate(r)
    
    print(f"value pos 1 : {value_pos1}")
    
    ################# Position2
    
    sigma2 = .2
    S02 = 1 
    K2 = 1
    pricer2 = CosPricer() 
    
    option2_maturity_date = trading_day + timedelta(int(365*.5))
    
    option2 = EuroPut(K1, option1_maturity_date)
    process2 = GBM(r, sigma2)
    
    units2 = 1
    
    position2 = PortfolioPosition(option2, process2, [S02], pricer2, units2, trading_day)
    
    value_pos2 = position2.evaluate(r)
    
    print(f"value pos 2 : {value_pos2}")
    
    
    ################# Portfolio 
    
    
    my_portfolio = Portfolio([position1, position2])

    porfolio_value = my_portfolio.evaluate(r)
    
    day = trading_day + timedelta(int(365*.25))
    n = 16
    
    VaR, ES, Loss_dist = my_portfolio.VaR_ES(.95, r, day, n)
    
    print(f"Portfolio value : {porfolio_value}")
    print(f"Portfolio VaR   : {VaR}")
    print(f"Portfolio ES    : {ES}")
    
    
    
        
        