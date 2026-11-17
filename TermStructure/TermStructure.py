# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:07:51 2026

@author: Nicola
"""



import numpy as np 
from options.DayCountConvetion_class import *
from datetime import datetime, timedelta

class DiscountFactor:
    """Discount Factor class P(t,T) = p"""
    def __init__(self, t : datetime, T : datetime, p : float, convention = None):
        
        if t>T:
            raise ValueError("The initial point t must be >= than the maturity T")
        
        self.t = t
        self.T = T 
        self.p = p 
        
        if convention==None:
            self.convention = Actual365()
        else:
            self.convention = convention
        
    def GetTenor(self):
        """compute the tenor of the discont factor in year fractions"""
        tenor = self.convention.year_fraction(self.t, self.T)
        
        return tenor 
    
    def GetRate(self):
        """compute r = -1/tenor * log(p)"""
        
        r = - 1 / self.GetTenor() * np.log(self.p)
        
        return r 
    
    # Operators overloading 
    
    def __str__(self):
        
        return f"t : {self.t}\nT : {self.T}\nP(t,T) : {self.p}"
    
    def __eq__(self, other):
        
        if self.t == other.t and self.T==other.T and np.isclose(self.p, other.p):
            return True 
        else:
            return False 
    
    def __gt__(self, other):
        
        if self.t != other.t:
            raise ValueError('Both discount factor must have the same initial time t to be compared')
        
        if self.T>other.T:
            return True  
        
        return False
        
    def __ge__(self, other):
        
        if self.t != other.t:
            raise ValueError('Both discount factor must have the same initial time t to be compared')
        
        if self.T>=other.T:
            return True  
        
        return False
    
    def __lt__(self, other):
        
        if self.t != other.t:
            raise ValueError('Both discount factor must have the same initial time t to be compared')
        
        if self.T<other.T:
            return True  
        
        return False
    
    def __le__(self, other):
        
        if self.t != other.t:
            raise ValueError('Both discount factor must have the same initial time t to be compared')
        
        if self.T<=other.T:
            return True  
        
        return False


##### The term structure class 

def ComputeDiscountFactor(t : float, T : float, r : float) -> float :
    
    return np.exp(-r * (T-t)) 


def ComputeForwardDiscountValue(DF1 : DiscountFactor, DF2 : DiscountFactor) -> float:
    
    """compute the discount factor value P(T1, T2) = P(t,T2)/P(t,T1), T2>T1""" 
    
    if DF1 > DF2:
        
        raise ValueError('DF1 must be < than DF2') 
        
    return DF2.p / DF1.p 

# def ComputeForwardDiscount(DF1 : DiscountFactor, DF2 : DiscountFactor) -> DiscountFactor:
    
#     # Get the new maturity (datetime object)
#     T = DF1.t + (DF2.T - DF1.T) 
    
#     # compute the value of the discount factor 
#     p = ComputeForwardDiscountValue(DF1, DF2)
    
#     return DiscountFactor(T, DF2.T, p)

def ComputeForwardDiscount(DF1, DF2):

    p = ComputeForwardDiscountValue(DF1, DF2)

    return DiscountFactor(DF1.T, DF2.T, p)

def ComputeDiscountRate(DF : DiscountFactor) -> float:
    
    tenor = DF.GetTenor()
    
    return -np.log(DF.p) / tenor


def interpolate_discout(Ti : datetime,  DF1 : DiscountFactor, DF2 : DiscountFactor, convention) -> DiscountFactor:
    
    t = DF1.t
    
    # Let's perform interpolation 
    
    tenor1 = DF1.GetTenor() 
    tenor2 = DF2.GetTenor() 
    
    r1 = DF1.GetRate()
    r2 = DF2.GetRate() 
    
    # compute the corresponding tenor with respect to Ti
    
    tenor_Ti = convention.year_fraction(t, Ti) 
    
    # interpolate r
    
    if Ti < DF1.T or Ti > DF2.T:
        
        m = (r2 - r1) / (tenor2 - tenor1)
        q = r2 - m * tenor2 
        
        r_interp = m * tenor_Ti + q 
    
    else:
    
        r_interp = np.interp(tenor_Ti, [tenor1, tenor2], [r1, r2])
    
    # compute the correponding discount factor p wrt r
    p = ComputeDiscountFactor(0, tenor_Ti, r_interp)
    
    # create the discuont factor instance 
    
    DF = DiscountFactor(t, Ti, p)
    
    return DF 


#######################

class TermStructure:
    
    def __init__(self, list_of_disc_factors, convention = None):
        
        self.discount_factors = sorted(list_of_disc_factors) 
        
        if convention==None:
            self.convention = Actual365()
        else:
            self.convention = convention 
    
    def __len__(self):
        
        return len(self.discount_factors)
        
    def GetRates(self) -> np.array:
        
        """2-d numpy array. first column is the tenor in year fraction and the second is the spot rate"""
        
        
        n = len(self.discount_factors)
        
        result = np.empty((n, 2)) # firts column : maturity - second column : rate 
        
        for i in range(0, n):
            
            result[i, 0] = self.discount_factors[i].GetTenor() 
            result[i, 1] = self.discount_factors[i].GetRate()
        
        return result 
    
    def GetFactor(self, Ti : datetime) -> DiscountFactor : 
        """return by the selected maturity Ti the corresponding discount factor, if missing it interpolates"""
        
    
        res = None # resulting variable (discount factor)
        
        DF1 = None # factor at maturity T_{i-1}
        DF2 = None # factor at maturity T_{i+1}
        
        t = self.discount_factors[0].t 
        n = len(self.discount_factors)
        
        # Step 1 check the boundaries (flat structure at the boundaries)
        
        if Ti < self.discount_factors[0].T:
            
            # return DiscountFactor(t, Ti, self.discount_factors[0].p)
            
            DF1 = self.discount_factors[0]
            DF2 = self.discount_factors[1]
            
            DF = interpolate_discout(Ti, DF1, DF2, self.convention)
            
            self.discount_factors.append(DF)
            
            return DF
        
        if Ti > self.discount_factors[-1].T:
            
            # return DiscountFactor(t, Ti, self.discount_factors[-1].p) 
            
            DF1 = self.discount_factors[-2]
            DF2 = self.discount_factors[-1]
            
            DF = interpolate_discout(Ti, DF1, DF2, self.convention)
            
            self.discount_factors.append(DF)
            
            return DF
        
        if Ti == self.discount_factors[-1].T:
            
            return DiscountFactor(t, Ti, self.discount_factors[-1].p) 
        
        for i in range(0, n-1):
            
            if self.discount_factors[i].T == Ti:
                
                res = self.discount_factors[i]
                
                return res 
            
            elif self.discount_factors[i].T < Ti and self.discount_factors[i+1].T > Ti:
                
                DF1 = self.discount_factors[i]
                DF2 = self.discount_factors[i+1]
                
                DF = interpolate_discout(Ti, DF1, DF2, self.convention)
                
                self.discount_factors.append(DF)
                
                self.discount_factors = sorted(self.discount_factors)
                
                return DF 
            
            else:
                pass
        
        # print('Something went wrong')
        return None 
    
    def GetForward(self, T1 : datetime, T2 : datetime) -> DiscountFactor :
        
        if T1>=T2:
            
            raise ValueError('Maturity T1 must < than T2')
            
        DF1 = self.GetFactor(T1)
        DF2 = self.GetFactor(T2)
        
        # if DF1==None or DF2==None:
            
        #     raise ValueError(f'Both DF1 and DF2 must be DiscountFactor type\nType DF1 : {type(DF1)}\nType DF2 : {type(DF2)}') 
        
        return ComputeForwardDiscount(DF1, DF2)
        

# if __name__=='__main__':
    
    # t = datetime.now() 
    
    # T1 = t + timedelta(days=365)
    # T2 = t + timedelta(days=365*3)
    
    # DF1 = DiscountFactor(t, T1, .95)
    # DF2 = DiscountFactor(t, T2, .85)
    
    # l = [DF2, DF1] 
    
    # TS = TermStructure(l)
    
    # DF12 = ComputeForwardDiscount(DF1, DF2)  
    
    # print()
    # print(DF12)
    
    # print()
    
    # DF12_from_TS = TS.GetForward(T1, T2)
    
    # print(DF12_from_TS)
    # print()
        
    
    # assert (DF12 == DF12_from_TS)
    
    # def test_get_forward():
        
    #     t = datetime.now() 
        
    #     T1 = t + timedelta(days=365)
    #     T2 = t + timedelta(days=365*3)
        
    #     DF1 = DiscountFactor(t, T1, .95)
    #     DF2 = DiscountFactor(t, T2, .85)
        
    #     l = [DF2, DF1] 
        
    #     TS = TermStructure(l)
        
    #     DF12 = ComputeForwardDiscount(DF1, DF2)  
        
    #     print()
    #     print(DF12.p)
        
    #     print()
        
    #     DF12_from_TS = TS.GetForward(T1, T2)
        
    #     print(DF12_from_TS.p)
    #     print()
        
    #     assert (DF12.t == DF12_from_TS.t)
    #     assert (DF12.T == DF12_from_TS.T)   
    #     np.testing.assert_allclose(DF12.p, DF12_from_TS.p, rtol=1e-2, atol=1e-2) 