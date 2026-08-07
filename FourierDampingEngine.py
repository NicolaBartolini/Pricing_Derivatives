# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:10:07 2026

@author: Nicola
"""

import numpy as np 
from scipy.integrate import quad 
from copy import copy 
from datetime import datetime 
from math import pi 


from EuropeanOption_class import EuropeanOption 
from process_class import process
from typing import Sequence


def damping_integrand(u : complex, option : EuropeanOption, process : process, X0 : Sequence[float], alpha : float, day=None) -> complex:
    
    if day==None:
        day = datetime.now() 
    
    T = option.get_tenor(day)
    
    S0 = X0[0]
    X1 = copy(X0)
    X1[0] = 1
    
    integrand = np.exp(-1j*u*np.log(S0)) * process.characteristic_fun(-(1j*alpha + u), X1, T) * option.damping_char_fun(u, alpha)
    
    return integrand 

def FourierDampingEngine(option : EuropeanOption, process : process, X0 : Sequence[float], r : float, alpha : float, day=None, a=0, b=250, n=5000) -> float:
    
    S0 = X0[0]
    
    if day==None:
        day = datetime.now() 
    
    T = option.get_tenor(day)
    
    fun = lambda u : damping_integrand(u, option, process, X0, alpha, day=day).real
    
    A = np.exp(-r*T) * S0**alpha / pi
    
    # n = int(max(N, b*N))
    
    integral = quad(fun, a, b, limit=n)[0]
    
    return A * integral


if __name__=="__main__":
    

    from datetime import timedelta
    from scipy.stats import norm

    from process_class import GBM
    from EuropeanOption_class import EuroCall


    def black_scholes_call(S0, K, r, sigma, T):
        d1 = (
            np.log(S0 / K)
            + (r + 0.5 * sigma**2) * T
        ) / (sigma * np.sqrt(T))

        d2 = d1 - sigma * np.sqrt(T)

        return (
            S0 * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )
    
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    alpha = 1.5
    sigma = .2 
    T = .5

    today = datetime(2026, 1, 1)
    maturity = today + timedelta(days=int(365 * T))

    option = EuroCall(K, maturity)
    process = GBM(r, sigma) 
    
    fourier_price = FourierDampingEngine(
        option=option,
        process=process,
        X0=[S0],
        r=r,
        alpha=alpha,
        day=today)

    bs_price = black_scholes_call(
        S0,
        K,
        r,
        sigma,
        T
    )