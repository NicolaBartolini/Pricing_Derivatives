# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 08:06:58 2026

@author: Nicola
"""

from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
import numpy as np 


#### Utils for COS pricing 

def CHI(k : float, a : float, b : float, c : float, d : float):
    # funzione Chi dell'articolo
    bma = b-a
    uu  = k * np.pi/bma
    cum = np.multiply(np.divide(1, (1 + np.power(uu,2))), (np.cos(uu * (d-a)) * np.exp(d) - np.cos(uu * (c-a)) * np.exp(c) + np.multiply(uu,np.sin(uu * (d-a))) * np.exp(d)-np.multiply(uu,np.sin(uu * (c-a))) * np.exp(c)))
    return cum

# Defintion of Psi (c,d) [Equation 23 Fang(2008)]
def PSI(k : float, a : float, b : float, c : float, d : float):
    bma    = b-a
    uu     = k * np.pi/bma
    uu[0]  = 1
    psi    = np.divide(1,uu) * ( np.sin(uu * (d-a)) - np.sin(uu * (c-a)) )
    psi[0] = d-c
    return psi

##################################
######## EUROPEAN OPTION CLASS
##################################

class EuropeanOption(ABC):
    
    @abstractmethod 
    def set_market_premium(self, P : float) -> None:
        pass 
    
    @abstractproperty 
    def get_premium(self) -> float :
        pass
    
    @abstractmethod 
    def get_tenor(self, date : datetime):
        pass 
    
    @abstractmethod 
    def payoff(self, S : np.array):
        pass
    
    @abstractmethod 
    def damping_char_fun(self, u : complex, alpha : float):
        pass
    
    # Defintion of U_k [Equation 29 Fang(2008)]
    @abstractmethod
    def UK(self, k : float, a : float, b : float):
        pass
  

class EuroCall(EuropeanOption):
    
    def __init__(self, strike : float, maturity_date : datetime):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium
    
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S):
        
        Pi = np.maximum(S-self.strike,0)
        
        return Pi 
    
    def damping_char_fun(self, u, alpha):
        
        K = self.strike 
        
        num = K**(1-alpha) * np.exp(1j*u*np.log(K)) 
        den = (1j*u - alpha) * (1j * u - alpha + 1.0) 
        
        return num/den 
    
    def UK(self, k, a, b):
        
        bma = b-a
        
        Uk = 2 / bma * (CHI(k,a,b,0,b) - PSI(k,a,b,0,b)) 
        
        return Uk
    

class EuroPut(EuropeanOption):
    
    def __init__(self, strike : float, maturity_date : datetime):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium 
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date 
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S):
        
        Pi = np.maximum(self.strike - S,0)
        
        return Pi 
    
    def damping_char_fun(self, u, alpha):
        
        K = self.strike 
        
        num = K**(1-alpha) * np.exp(1j*u*np.log(K)) 
        den = (1j*u - alpha) * (1j * u - alpha + 1.0) 
        
        return num/den 
    
    def UK(self, k, a, b):
        
        bma = b-a
        
        Uk = 2 / bma * (-CHI(k,a,b,a,0) + PSI(k,a,b,a,0) )
        
        return Uk

class EuroCallDigital(EuropeanOption):
    # This option pays 1 if the underlying is greater than the strike and 0 otherwise
    
    def __init__(self, strike : float, maturity_date : datetime):
        
        self.strike = strike 
        self.maturity_date = maturity_date 
        
        self.market_premium = np.nan 
        
    def set_market_premium(self, P):
        
        self.market_premium = P 
        
    def get_premium(self) -> float :
        return self.market_premium
    
    def get_tenor(self, date):
        
        diff = self.maturity_date - date
        
        tenor = diff.days/365 
        
        return tenor  
    
    def payoff(self, S):
        
        return np.where(S > self.strike, 1.0, .0)
            
    def damping_char_fun(self, u, alpha):
        
        K = self.strike 
        
        num = -K**(1j*u -alpha)
        den = (1j*u -alpha) 
        
        return num/den 
    
    def UK(self, k, a, b):
        
        bma = b-a
        
        Uk = 2 / bma * PSI(k,a,b,0,b) * 1/self.strike
        
        return Uk
    

class EuroPutDigital(EuropeanOption):
    # This option pays 1 if the underlying is below the strike
    # and 0 otherwise

    def __init__(self, strike: float, maturity_date: datetime):

        self.strike = strike
        self.maturity_date = maturity_date

        self.market_premium = np.nan

    def set_market_premium(self, P):
        self.market_premium = P

    def get_premium(self) -> float:
        return self.market_premium

    def get_tenor(self, date):
        diff = self.maturity_date - date
        tenor = diff.days / 365

        return tenor

    def payoff(self, S):

        return np.where(S < self.strike, 1.0, 0.0)

    def damping_char_fun(self, u, alpha):

        # K = self.strike

        # num = K**(1 - alpha) * np.exp(1j * u * np.log(K))
        # den = (1j * u - alpha)
        
        K = self.strike 
        
        # num = -K**(1j*u -alpha)
        num = K**(1j*u -alpha)
        den = (1j*u -alpha) 

        return num / den

    def UK(self, k, a, b):

        bma = b - a

        Uk = 2 / bma * PSI(k, a, b, a, 0) * 1/self.strike

        return Uk


class EuroCallAssetOrNothing(EuropeanOption):
    # Pays S_T if S_T > K, and 0 otherwise

    def __init__(self, strike: float, maturity_date: datetime):

        self.strike = strike
        self.maturity_date = maturity_date

        self.market_premium = np.nan

    def set_market_premium(self, P):
        self.market_premium = P

    def get_premium(self) -> float:
        return self.market_premium

    def get_tenor(self, date):
        diff = self.maturity_date - date
        tenor = diff.days / 365

        return tenor

    def payoff(self, S):

        return np.where(S > self.strike, S, 0.0)

    def damping_char_fun(self, u, alpha):

        K = self.strike

        num = - K**(1 + 1j * (1j*alpha + u))
        den = 1 + 1j * (1j*alpha + u)

        return num / den

    def UK(self, k, a, b):
        
        pass
        # bma = b - a

        # Uk = 2 / bma * CHI(k, a, b, self.strike, b) * 1/self.strike

        # return Uk


class EuroPutAssetOrNothing(EuropeanOption):
    # Pays S_T if S_T < K, and 0 otherwise

    def __init__(self, strike: float, maturity_date: datetime):

        self.strike = strike
        self.maturity_date = maturity_date

        self.market_premium = np.nan

    def set_market_premium(self, P):
        self.market_premium = P

    def get_premium(self) -> float:
        return self.market_premium

    def get_tenor(self, date):
        diff = self.maturity_date - date
        tenor = diff.days / 365

        return tenor

    def payoff(self, S):

        return np.where(S < self.strike, S, 0.0)

    def damping_char_fun(self, u, alpha):

        K = self.strike

        num = - K**(1 + 1j * (1j*alpha + u))
        den = 1 + 1j * (1j*alpha + u)

        return -num / den

    def UK(self, k, a, b):
        
        pass
        # bma = b - a

        # Uk = 2 / bma * CHI(k, a, b, a, self.strike) * 1/self.strike

        # return Uk