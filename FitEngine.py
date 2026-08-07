# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:44:10 2026

@author: Nicola
"""

import numpy as np 
from scipy.optimize import minimize, Bounds 
from process_class import GBM, process 
from EuropeanOption_class import EuropeanOption
from typing import Sequence
from datetime import datetime
from FourierDampingEngine import FourierDampingEngine
from CosPricingEngine import CosPricingEngine



def objective_function(params : np.array, process : process, OPTIONS : Sequence[EuropeanOption], X0 : np.array, r : float, date : datetime):
    
    """
    params (np.array) : parameters to fit if the process to be fitted is of the 
                        Heston type (stochastic vol), the last element of 
                        params will be used as initial point for the vol and 
                        it also will be estimated
    
    process : the gbm process 
    options (list[EuropeanOptions]) : list of all the European options  
    r (float) : risk-free rate
    date : trading day of the evaluation (used to compute the tenor of the option)
    """
    
    res = 0 
    
    if len(X0)>1:
        
        # parameters = np.concatenate(([r], np.asarray(params, dtype=float)))
        
        process.set_parameters(np.hstack((np.array([r]), params[:-1])))
        
        X1 = X0.copy()
        X1[1] = params[-1] # The initial value of the vol
    
    else:
        # print("I'm here")
        in_params = np.hstack((np.array([r]), params))
        
        # print(in_params)
        
        process.set_parameters(in_params)
        X1 = X0.copy()
    
    # print(X1)
    # print(in_params)
    # print(process.mu)
    # print(process.sigma)
    # print()
    
     
    for option in OPTIONS:
        
        price = CosPricingEngine(option, process, X1, r)
        
        res += (option.get_premium() - price)**2 
    
    # print(res)
    
    return res 


def fit_process(params : np.array, process:process, OPTIONS : Sequence[EuropeanOption], X0 : np.array, r : float, date : datetime, lb=0, ub=3) -> tuple[process, np.array]:
    
    bounds = Bounds(lb, ub)
    
    res = minimize(objective_function, params, args=(process, OPTIONS, X0, r, date),
                   method='L-BFGS-B', bounds=bounds) 
    
    process.set_parameters(np.hstack((r, res.x))) 
    
    return (process, res)


if __name__=='__main__':
    
    np.random.seed(42)
    
    from EuropeanOption_class import EuroCall, EuroPut 
    from process_class import GBM, Heston93, Bates
    
    maturities = [datetime(2026, 9, 30), datetime(2026, 10, 30), datetime(2026, 11, 30)]
    
    S0 = 1 
    sigma = .2 
    v0 = .01
    r = .035 
    
    # X0 = np.array([S0])
    
    X0 = np.array([S0, v0])
    
    kappa = 1.3 
    theta = .015 
    eta = .05
    
    rho = -.07
    
    # process_true = GBM(mu=r, sigma=sigma)
    process_true = Heston93(r, kappa, theta, eta, rho)
    
    strikes = np.array([.8, .9, 1, 1.1, 1.2]) 
    
    OPTIONS = []  
    
    today = datetime.now()
    
    for maturity in maturities:

        for strike in strikes:

            ############################################################
            # Call
            ############################################################

            call = EuroCall(strike, maturity)

            price = FourierDampingEngine(option=call, process=process_true, X0=X0, r=r, alpha=1.5, day=today)

            call.set_market_premium(price)

            OPTIONS.append(call)

            ############################################################
            # Put
            ############################################################

            put = EuroPut(strike, maturity)

            price = FourierDampingEngine(option=put, process=process_true, X0=X0, r=r, alpha=-1.5, day=today)

            put.set_market_premium(price)

            OPTIONS.append(put)

    print(f"Number of options: {len(OPTIONS)}")
    
    ####################################################################
    # Initial guess
    ####################################################################

    # process_guess = GBM(
    #     mu=r,
    #     sigma=0.35
    # )

    # initial_guess = np.array([0.35]) 
    
    process_guess = Heston93() 
    initial_guess = np.random.uniform(1e-4, .99, 5)
    initial_guess[-2] = 0 
    
    ####################################################################
    # Calibration
    ####################################################################

    res = fit_process(
        params=initial_guess,
        process=process_guess,
        OPTIONS=OPTIONS,
        X0=X0,
        r=r,
        # alpha=1.5,
        date=today,
        lb=[1e-4,1e-4,1e-4,-1, 1e-4],
        ub=[5,5,5,1, 1]
    )

    ####################################################################
    # Results
    ####################################################################

    # print()
    # print("==========================================")
    # print(f"True sigma      : {sigma:.6f}")
    # print(f"Estimated sigma : {sigma_hat[0]:.6f}")
    # print("==========================================")

    # print(res)
    
    new_OPTIONS = []  
    
    today = datetime.now()
    
    for maturity in maturities:

        for strike in strikes:

            ############################################################
            # Call
            ############################################################

            call = EuroCall(strike, maturity)

            price = FourierDampingEngine(option=call, process=process_guess, X0=X0, r=r, alpha=1.5, day=today)

            call.set_market_premium(price)

            new_OPTIONS.append(call)

            ############################################################
            # Put
            ############################################################

            put = EuroPut(strike, maturity)

            price = FourierDampingEngine(option=put, process=process_true, X0=X0, r=r, alpha=-1.5, day=today)

            put.set_market_premium(price)

            new_OPTIONS.append(put) 
            
    
    for i in range(len(OPTIONS)):
        
        p1 = OPTIONS[i].get_premium()
        p2 = new_OPTIONS[i].get_premium() 
        print(f"{p1} \t:\t {p2} \t:\t {abs(p1-p2)}")
     
    
    