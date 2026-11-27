# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:21:07 2026

@author: Nicola
"""

from instruments.bonds.bonds import *  
from process.additive_process import * 
from datetime import datetime 
import numpy as np 



##### file with analytic formulas for bond pricing 

def PriceZCB(bond2price, process : additive_process, X0 : float, current_day : datetime) -> float:
    
    """Pricing function zcb"""
    
    T = bond2price.get_tenor(current_day) 
    
    affine_coeff = process.affine_coefficient(T, 0) 
    
    A = affine_coeff['A']
    B = affine_coeff['B']
    
    zcb_price = np.exp(A - B*X0)
    
    return zcb_price 


def PriceCB(bond2price, process : additive_process, X0 : float, current_day : datetime) -> float:
    
    T = bond2price.get_tenor(current_day) 
    
    affine_coeff = process.affine_coefficient(T, 0) 
    
    A = affine_coeff['A']
    B = affine_coeff['B']
    
    zcb_part = np.exp(A - B*X0)
    
    # Now we compute the part of the price related to the coupons 
    
    coupon_part = 0 
    c = bond2price.coupon_rate 
    
    payments_grid_tenor = bond2price.get_tenor_coupon_payments(current_day)
    
    if len(payments_grid_tenor)==0:
        return zcb_part 
    
    for Ti in payments_grid_tenor:
        
        affine_coeff = process.affine_coefficient(Ti, 0) 
        
        Ai = affine_coeff['A']
        Bi = affine_coeff['B']
        
        coupon_part += c * np.exp(Ai - Bi*X0) 
    
    return zcb_part + coupon_part
    
    


