# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:04:18 2026

@author: Nicola
"""

import numpy as np 
from abc import ABC, abstractmethod 
from typing import Any, Callable, Optional, Union, Sequence, Iterable
from scipy.stats import ncx2

class additive_process(ABC):
    
    # @abstractmethod 
    # def set_parameters(self, parameters : np.array) -> None :
    #     pass
    
    
    @abstractmethod 
    def simulate(self, X0 : Sequence[float], T : float, n_steps : int, n=10) -> np.ndarray | tuple[np.ndarray, np.ndarray] :
        # this method simulates a stochastic process
        pass 
    
    @abstractmethod
    def affine_coefficient(self, X0 : Sequence[float], T : float, t : float) -> dict:
        # this method computes the affine coefficient of E[exp^{-int_t^T x_u du} | F_t], i.e. exp^{A(t,T) + B(t,T)x_t}
        pass 
    
    # @abstractmethod
    # def characteristic_fun(self, u : complex, X0 : Sequence[float], T : float, t : float) -> complex:
    #     # this method computes the characteristic function of a stochastic process
    #     pass 
    

class OrnsteinUhlenbeck(additive_process): 
    
    def __init__(self, kappa, theta, sigma):
        
        if kappa <= 0 or sigma <= 0:
            raise ValueError("kappa, and sigma must be positive.")
        
        self.kappa = kappa 
        self.theta = theta 
        self.sigma = sigma 
    
    def simulate(self, X0 : Sequence[float], T : float, n_steps : int, n=10) -> np.ndarray:
        # this method simulates a stochastic process
        
        N = 1<<n # total number of trajectories 
        
        # allocate memory for the simulated trajectories
        Xt = np.empty((n_steps+1, N)) # store the trajectories 
        Xt[0] = X0[0] 
        
        # compute the time step
        dt = T/n_steps
        v = np.sqrt(dt) * self.sigma
        
        for i in range(1, n_steps+1):
            
            # generate gaussian noise with antithetic variate
            if n>0:
                
                eps = np.random.normal(0,1,(N//2))
                eps = np.hstack((eps, -eps))
            
            else:
                eps = np.random.normal(0,1) 
            # compute the trajectory with the Euler scheme
            Xt[i] = Xt[i-1] + self.kappa * (self.theta - Xt[i-1]) * dt + v * eps 
        
        return Xt
            
            
    
    def affine_coefficient(self, X0 : Sequence[float], T : float, t : float) -> dict:
        # this method computes the affine coefficient of E[exp^{-int_t^T x_u du} | F_t], i.e. exp^{A(t,T) + B(t,T)x_t}
        
        B = (1.0 - np.exp(-self.kappa * (T - t))) / self.kappa 
        A = (self.theta - self.sigma**2 / (2.0 * self.kappa**2)) * (B - (T - t)) - (self.sigma**2 / (4.0 * self.kappa)) * B**2 
        
        return {'A' : A, 'B' : B} 


class CIR(additive_process):

    def __init__(self, kappa, theta, sigma):
        
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            raise ValueError("kappa, theta and sigma must be positive.")

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def set_parameters(self, parameters: np.ndarray) -> None:

        self.kappa, self.theta, self.sigma = parameters

    def simulate(self, X0: Sequence[float], T: float, n_steps: int, n=10) -> np.ndarray:
        """
        Simulate CIR trajectories using exact transitions
        from the non-central chi-square distribution.

        Returns
        -------
        np.ndarray
            Array with shape (n_steps + 1, 2**n).
        """
        
        if T < 0 or n_steps <= 0:
            raise ValueError("T must be non-negative and n_steps positive.")

        if n < 0:
            raise ValueError("n must be non-negative.")

        N = 1 << n
        dt = T / n_steps

        Xt = np.empty((n_steps + 1, N))
        Xt[0, :] = X0[0]

        if Xt[0, 0] < 0:
            raise ValueError("The initial CIR value must be non-negative.")

        exp_kdt = np.exp(-self.kappa * dt)
        
        c = (self.sigma**2 * (-np.expm1(-self.kappa * dt)) / (4.0 * self.kappa))

        degrees_freedom = ( 4.0 * self.kappa * self.theta / self.sigma**2)

        for i in range(1, n_steps + 1):

            noncentrality = (exp_kdt * Xt[i - 1] / c)

            Xt[i] = c * ncx2.rvs(df=degrees_freedom, nc=noncentrality, size=N)

        return Xt

    def affine_coefficient(self, X0: Sequence[float], T: float, t: float) -> dict:
        """
        Compute A and B such that

        E[exp(-u * integral_t^T X_s ds) | F_t]
            = exp(A + B * X_t).

        Here u is the Laplace-transform argument.
        The standard CIR zero-coupon bond coefficients
        correspond to u = 1.
        """

        if T < t:
            raise ValueError("T must be greater than or equal to t.")

        tau = T - t

        if tau == 0:
            return {"A": 0.0, "B": 0.0}

        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma

        gamma = np.sqrt(kappa**2 + 2.0 * sigma**2)

        exp_term = np.exp(-gamma * tau)

        denominator = ((gamma + kappa) * (1.0 - exp_term) + 2.0 * gamma * exp_term)

        B = -2.0 * (1.0 - exp_term) / denominator

        A = (2.0 * kappa * theta / sigma**2 * (np.log(2.0 * gamma) + (kappa - gamma) * tau / 2.0 - np.log(denominator)))

        return {"A": A, "B": B}

        
    