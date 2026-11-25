# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:29:06 2026

@author: Nicola
"""

import numpy as np 

#### Implied volatility formulas 

#### Imp vol for Hagan 2002 

def z_hagan2002(alpha0, F0, K, nu, beta):
    
    # alpha0 : exit point of the vol process 
    # F0     : exit point of the Libor process 
    # K      : strike of the option 
    # nu     : vol-vol 
    # beta   : elasticity parameter
    
    z = nu/alpha0 * (F0*K)**(.5 * (1-beta)) * np.log(F0/K)
    
    return z 

def x_hagan2002(alpha0, F0, K, nu, beta, rho):
    
    z = z_hagan2002(alpha0, F0, K, nu, beta) 
    
    x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    
    return x 


def imp_atm_sigma_hagan2002(alpha0, F0, nu, beta, rho, T):
    
    A = alpha0/F0**(1-beta) 
    
    B = (1-beta)**2 / 24 * (alpha0**2) / (F0**(2 - 2*beta))
    
    C = .25 * (rho*alpha0*nu*beta) / (F0**(1-beta)) 
    
    D = nu**2 * (2 - 3*rho**2)/24 
    
    result = A * (1 + (B + C + D)*T)
    
    return result 

def imp_vol_sigma_hagan2002(alpha0, F0, K, nu, beta, rho, T):

    A = (F0 * K)**(.5 * (1 -beta)) 
    B = ((1-beta)**2)/24 * np.log(F0/K)**2 
    C = (1-beta)**4 / (1920) * np.log(F0/K)**4 

    D = (1-beta)**2 / 24 * alpha0**2 / (F0*K)**(1-beta)
    E = .25 * (rho*beta*nu*alpha0) / (F0*K)**(.5 * (1-beta)) 
    F = (2 -3 * rho**2)*nu**2 / 24 

    x = x_hagan2002(alpha0, F0, K, nu, beta, rho)
    z = z_hagan2002(alpha0, F0, K, nu, beta) 

    result = alpha0 / (A * (1 + B + C)) * (z/x) * (1 + (D + E + F) * T)

    return result
