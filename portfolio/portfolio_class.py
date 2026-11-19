# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:32:17 2026

@author: Nicola
"""

import sys 
sys.path.append(r'C:\Users\Nicola\Documents\Python_Scripts\0_finance')

from options.EuropeanOption_class import *
from options.AsianOption_class import * 
from options.AmericanOption_class import * 
from options.BarrierOption_class import * 
from pricing.PricingEngine_class import * 
from pricing.DeltaHedgingEngine import compute_Delta

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
    
    def simulate(self, r, day, n, X0=None):
        
        N = 1<<n # number of Monte-Carlo simulations
        n_steps = (day - self.trading_day).days
        t1 = n_steps/365
        
        # Simulate the process underying up to the VaR evaluation date
        if X0==None:
            sim = self.process.simulate(self.X0, t1, n_steps, n)
        else:
            sim = self.process.simulate(X0, t1, n_steps, n)
        
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



# class Portfolio:
    
#     def __init__(self, assets : list[PortfolioPosition], multivariate_model_underP=np.nan):
        
#         self.assets = assets 
#         self.multi_model = multivariate_model_underP
#         self.trading_day = datetime.now()
        
#     def evaluate(self, r : float):
        
#         res = 0 
        
#         for asset in self.assets:
            
#             res += asset.evaluate(r) 
        
#         return res 
    
#     def VaR_ES(self, q, r, day, n):
        
#         portfolio_value_at_t0 = self.evaluate(r)
#         portfolio_value_at_t1 = np.zeros(1<<n)
        
#         ##### let's simulate all the underlying assets with the multivariate model up to t1
        
#         N = 1<<n # number of Monte-Carlo simulations
#         n_steps = (day - self.trading_day).days
#         t1 = n_steps/365
        
#         St = self.multi_model.simulate(S0, t1, n_steps, N)[:, -1, :] # each row is a different MC trj and each column is a different asset
        
#         for j in range(1, N):
#             P = 0
#             for i,asset in enumerate(self.assets):
                
#                 P += asset.simulate(r, day, n, St[j, i])
            
#             portfolio_value_at_t1[j] = P
        
#         Loss = portfolio_value_at_t0 - portfolio_value_at_t1
        
#         VaR = np.quantile(Loss, q)
#         ES = np.mean(Loss[Loss>= VaR])
        
#         return VaR, ES, Loss


if __name__=='__main__':
    
    from datetime import timedelta
    from process.process_class import GBM
    from process.multivariate_process import MultiGBM
    
    r = .025
    trading_day = datetime.now()
    
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
    
    
    # ################# Portfolio
    
    # rho = -.1
    
    # SIGMA = np.array([[sigma1**2,         sigma1*sigma2*rho], 
    #                   [sigma1*sigma2*rho, sigma2**2]])
    
    # MU = np.array([0.01, 0.02]) 
    
    # S0 = np.array([S01, S02])
    
    # multimodel = MultiGBM(MU, SIGMA)
        
    # my_portfolio = Portfolio([position1, position2], multimodel)

    # porfolio_value = my_portfolio.evaluate(r)
    
    # day = trading_day + timedelta(int(365*.25))
    # n = 16
    
    # VaR, ES, Loss_dist = my_portfolio.VaR_ES(.95, r, day, n)
    
    # print(f"Portfolio value : {porfolio_value}")
    # print(f"Portfolio VaR   : {VaR}")
    # print(f"Portfolio ES    : {ES}")
    
    
    
        
        