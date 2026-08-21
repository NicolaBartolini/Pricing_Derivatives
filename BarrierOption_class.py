# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:20:10 2026

@author: Nicola
"""

from abc import ABC, abstractmethod 
import numpy as np 
from datetime import datetime 
from numba import njit 

@njit()
def check_lower_barrier_touch(x : np.array, L : float):
    
    result = 0
    
    if np.min(x)<=L:
        result = 1
    
    return result 

@njit()
def check_upper_barrier_touch(x : np.array, L : float):
    
    result = 0
    
    if np.max(x)>=L:
        result = 1
    
    return result 



##################


class BarrierOption(ABC):
    
    @abstractmethod 
    def set_market_premium(self, P : float) -> None:
        pass 
    
    @abstractmethod 
    def get_premium(self) -> float :
        pass 
    
    @abstractmethod
    def get_barrier(self) -> float :
        pass
    
    @abstractmethod 
    def get_tenor(self, date : datetime):
        pass 
    
    @abstractmethod 
    def payoff(self, S : np.array, nColumns : int):
        pass   
    

class DownInCall(BarrierOption):
    
    def __init__(self, strike : float, barrier : float, maturity_date):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        self.barrier = barrier
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def get_barrier(self) -> float :
        return self.barrier
    
    def payoff(self, S, nColumns):
        
        activation = np.zeros(nColumns)
        
        for i in range(nColumns):
            
            activation[i] = check_lower_barrier_touch(S[:,i], self.barrier)
        
        return np.maximum(S[-1] * activation - self.strike, 0)


class DownOutCall(BarrierOption):
    
    def __init__(self, strike : float, barrier : float, maturity_date):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        self.barrier = barrier
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
    
    def get_barrier(self) -> float :
        return self.barrier
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S, nColumns):
        
        activation = np.zeros(nColumns)
        
        for i in range(nColumns):
            
            activation[i] = check_lower_barrier_touch(S[:,i], self.barrier)
        
        return np.maximum(S[-1] * (1-activation) - self.strike, 0)




class UpInCall(BarrierOption):
    
    def __init__(self, strike : float, barrier : float, maturity_date):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        self.barrier = barrier
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_barrier(self) -> float :
        return self.barrier
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S, nColumns):
        
        activation = np.zeros(nColumns)
        
        for i in range(nColumns):
            
            activation[i] = check_upper_barrier_touch(S[:,i], self.barrier)
        
        return np.maximum(S[-1] * activation - self.strike, 0)

class UpOutCall(BarrierOption):
    
    def __init__(self, strike : float, barrier : float, maturity_date):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        self.barrier = barrier
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def get_barrier(self) -> float :
        return self.barrier
    
    def payoff(self, S, nColumns):
        
        activation = np.zeros(nColumns)
        
        for i in range(nColumns):
            
            activation[i] = check_upper_barrier_touch(S[:,i], self.barrier)
        
        return np.maximum(S[-1] * (1-activation) - self.strike, 0)
    

############## put options 

class DownInPut(BarrierOption):
    
    def __init__(self, strike : float, barrier : float, maturity_date):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        self.barrier = barrier
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def get_barrier(self) -> float :
        return self.barrier
    
    def payoff(self, S, nColumns):
        
        activation = np.zeros(nColumns)
        
        for i in range(nColumns):
            
            activation[i] = check_lower_barrier_touch(S[:,i], self.barrier)
        
        return np.maximum(self.strike - S[-1] * activation, 0)



class DownOutPut(BarrierOption):
    
    def __init__(self, strike : float, barrier : float, maturity_date):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        self.barrier = barrier
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
    
    def get_barrier(self) -> float :
        return self.barrier
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S, nColumns):
        
        activation = np.zeros(nColumns)
        
        for i in range(nColumns):
            
            activation[i] = check_lower_barrier_touch(S[:,i], self.barrier)
        
        return np.maximum(self.strike - S[-1] * (1-activation), 0)


class UpInPut(BarrierOption):
    
    def __init__(self, strike : float, barrier : float, maturity_date):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        self.barrier = barrier
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_barrier(self) -> float :
        return self.barrier
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S, nColumns):
        
        activation = np.zeros(nColumns)
        
        for i in range(nColumns):
            
            activation[i] = check_upper_barrier_touch(S[:,i], self.barrier)
        
        return np.maximum(self.strike - S[-1] * activation, 0)

class UpOutPut(BarrierOption):
    
    def __init__(self, strike : float, barrier : float, maturity_date):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        self.barrier = barrier
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def get_barrier(self) -> float :
        return self.barrier
    
    def payoff(self, S, nColumns):
        
        activation = np.zeros(nColumns)
        
        for i in range(nColumns):
            
            activation[i] = check_upper_barrier_touch(S[:,i], self.barrier)
        
        return np.maximum(self.strike - S[-1] * (1-activation), 0)