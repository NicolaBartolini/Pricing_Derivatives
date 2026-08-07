# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:13:09 2026

@author: Nicola
"""

from AmericanOption_class import AmericanOption 
from process_class import process 

import numpy as np 
from datetime import datetime, timedelta

import scipy.special
from scipy.special import laguerre

def OLS(X,Y):
    return np.linalg.inv(X.T@X)@X.T@Y

def simulate_gbm(S0, r, sigma, T, N, M):
    dt = T / N
    Z = np.random.randn(N, M)
    S = np.zeros((N+1, M))
    S[0, :] = S0
    for t in range(1, N+1):
        S[t, :] = S[t-1, :] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1, :])
    return S

def LaguerreBase(degree, x):
    
    result = [] 
    
    for i in range(0, degree+1):
        result.append(laguerre(i)(x))
    
    return np.array(result)

def TaylorPoly(degree, x):
    
    result = 0
    
    for i in range(0, degree+1):
        if i==0 or i==1:
            result += x**i 
        else:
            result += x**i/scipy.special.gamma(i+1)
    
    return result 

def TaylorBase(degree, x):
    
    result = [] 
    
    for i in range(0, degree+1):
        result.append(TaylorPoly(i, x))
    
    return np.array(result)

def LongstaffSchwartzEngine(process:process, option:AmericanOption, X0 : np.array, today:datetime, r : float, N=10, degree=3, basis='Laguerre'):
    
    ########
    
    T = option.get_tenor(today)
    n_days = option.get_business_number_of_days(today)
        
    #### Let's simulate the process 
    
    sim = process.simulate( X0, T, n_days, N)
    
    if isinstance(sim, tuple):
        S = sim[0]      # Heston: (S, V)
    else:
        S = sim         # GBM: S
    
    exercise_date = option.get_payoff_grid(today)
    
    if len(exercise_date)!=len(S):
        raise ValueError('length of exercise_date must be equal to length of S')
        
    N = len(exercise_date)
    # dt = T / N
    payoff = option.payoff(S)
    
    # Discounted future cashflows
    cashflow = payoff[-1, :]
    
    for t in range(N-1, 0, -1):
        
        dt = exercise_date[t] - exercise_date[t-1]
        # print(dt)
        ITM = payoff[t, :] > 0 # array where 
        X = S[t, ITM]
        Y = cashflow[ITM] * np.exp(-r * dt)
        
        if basis=='Laguerre':
        # Basis functions: Laguerre polynomials
            A = LaguerreBase(degree, X).T
        elif basis=='Taylor':
            A = TaylorBase(degree, X).T
        else:
            raise ValueError('Wrong basis. Only Laguerre or Taylor')
        
        beta = OLS(A, Y)
        C_t = A @ beta # E[Y|X]
        
        exercise = payoff[t, ITM] > C_t # where E[Y|X] > payoff at time t
        
        cashflow[ITM] = np.where(exercise, payoff[t, ITM], cashflow[ITM] * np.exp(-r * dt))
    
    dt = exercise_date[1] - exercise_date[0]
    
    return np.mean(cashflow * np.exp(-r * dt)) 


if __name__=='__main__':
    
    from process_class import GBM
    from AmericanOption_class import AmericanPut, AmericanCall

        
    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=365)
    
    ###################################################
    # Underlying process
    ###################################################
    
    process = GBM(
        mu=0.05,
        sigma=0.20
    )
    
    ###################################################
    # American option
    ###################################################
    
    option = AmericanPut(
        strike=100.0,
        maturity_date=maturity
    )
    
    ###################################################
    # Initial state
    ###################################################
    
    X0 = [100.0]
    
    ###################################################
    # Pricing
    ###################################################
    
    price = LongstaffSchwartzEngine(
        process=process,
        option=option,
        X0=X0,
        today=today,
        r=0.05,
        N=12,                 # 2^12 Monte Carlo paths
        degree=3,
        basis="Laguerre"
    )
    
    option2 = AmericanCall( strike=100.0, maturity_date=maturity)
    
    price2 = LongstaffSchwartzEngine(
        process=process,
        option=option2,
        X0=X0,
        today=today,
        r=0.05,
        N=12,                 # 2^12 Monte Carlo paths
        degree=3,
        basis="Laguerre"
    )
    
    print(f"American put  price = {price:.6f}")
    print(f"American call price = {price2:.6f}")