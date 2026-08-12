# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:37:23 2026

@author: Nicola
"""

from abc import ABC, abstractmethod 
import numpy as np  
from scipy.stats import norm 

import numpy.typing as npt

from typing import Any, Callable, Optional, Union, Sequence, Iterable

def cir_conditional_mean(x0 : float, k : float, theta : float, sigma : float, dt : float) -> float:
    
    return x0 * np.exp(-k * dt) + theta * (1 - np.exp(-k*dt))

def cir_conditional_variance(x0 : float, k : float, theta : float, sigma : float, dt : float) -> float:
    
    return x0 * (sigma**2)/k * (np.exp(-k*dt) - np.exp(-2*k*dt)) + (theta*sigma**2)/(2*k) * (1 - np.exp(-k*dt))**2;


class process(ABC):
    
    @abstractmethod 
    def set_parameters(self, parameters : np.array) -> None :
        pass
    
    
    @abstractmethod 
    def simulate(self, X0 : Sequence[float], T : float, n_steps : int, n=10) -> np.ndarray | tuple[np.ndarray, np.ndarray] :
        # this method simulates a stochastic process
        pass 
    
    @abstractmethod
    def characteristic_fun(self, u : complex, X0 : Sequence[float], T : float, t : float) -> complex:
        # this method computes the characteristic function of a stochastic process
        pass
    
    @abstractmethod
    def cos_cumulants(self, X0 : Sequence[float], T : float) -> tuple[float, float, float] :
        """
        Returns (c1, c2, c4)
        """
        pass


class GBM(process):
    # GBM = Geometric Brownian Motion
    def __init__(self, mu=np.nan, sigma = np.nan):
        # mu = drift 
        # sigma = volatility
        self.mu = mu 
        self.sigma = sigma
        
    def set_parameters(self, parameters):
        self.mu = parameters[0]
        self.sigma = parameters[1]
        
    def simulate(self, X0, T, n_steps, n):
        
        N = 1<<n # total number of Monte-Carlo simulations (2^n, usefull for antithetic variate)
        S0 = X0[0] # getting the initial value 
        St = np.empty((n_steps+1, N)) # memory pre-allocation of the simulation
        St[0] = S0 # initializing the first value 
        
        dt = T/n_steps # time increment
        drift = (self.mu -.5*self.sigma**2) * dt # drift part from the solution of the GBM SDE
        
        vol = self.sigma*np.sqrt(dt) # actual standard deviation of the process
        
        for i in range(1, n_steps+1):
            # Compute via antithetic variate the Gaussian noise
            if n>0:
                eps = np.random.normal(0,1,(1, int(N/2)))
                eps = np.hstack((eps, -eps))
            else:
                eps = np.random.normal(0,1) 
            
            St[i] = St[i-1] * np.exp(drift + vol*eps)
        
        return St 
    
    def characteristic_fun(self, u, X0, T):
        
        S0 = X0[0]
        
        m = (np.log(S0) + (self.mu - 0.5*self.sigma**2)*T)
        
        expon = 1j*u*m - .5 * self.sigma**2 * u**2 *T
        
        return np.exp(expon) 
    
    def cos_cumulants(self, X0, T):

        S0 = X0[0]

        c1 = np.log(S0) + (self.mu - 0.5*self.sigma**2)*T
        c2 = self.sigma**2*T
        c4 = 0.0

        return c1, c2, c4
        
        
class Heston93(process):
    
    def __init__(self, mu=np.nan, kappa=np.nan, theta=np.nan, eta=np.nan, rho=np.nan, gamma1=.5, gamma2=.5):
        
        self.mu = mu
        self.kappa = kappa 
        self.theta = theta
        self.eta = eta 
        self.rho = rho
        
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    
    def set_parameters(self, parameters : np.array) -> None :
        
        self.mu = parameters[0]
        self.kappa = parameters[1] 
        self.theta = parameters[2]
        self.eta = parameters[3] 
        self.rho = parameters[4]
    
    def simulate(self, X0, T, n_steps, n=10):
        # this method simulates a stochastic process
        
        S0 = X0[0]
        v0 = X0[1]
    
        mu = self.mu
        kappa = self.kappa
        theta = self.theta
        eta = self.eta
        rho = self.rho
    
        N_paths = 1 << n
        dt = T / n_steps
    
        S = np.empty((n_steps + 1, N_paths))
        V = np.empty((n_steps + 1, N_paths))

        S[0] = S0
        V[0] = v0 
        
        noise = np.random.uniform(0,1,(n_steps, 2*N_paths)); # generating the noise from a uniform distribution 
        
        uniform_sampling = noise[:,0:N_paths];
        
        X_noise = noise[:,N_paths:]; # taking the first uniform observations for generating the noise for the volatility vol_trj
        
        for i in np.arange(1, n_steps+1): # iterating for each step 
            
            m = cir_conditional_mean(V[i-1], kappa, theta, eta, dt);
            s_square = cir_conditional_variance(V[i-1], kappa, theta, eta, dt);
           
            psi = s_square/(m**2);
           
            psi_normal = psi[psi<=1.5];
            m_normal = m[psi<=1.5];
            
            b_square = 2/psi_normal - 1 + np.sqrt(2/psi_normal) * np.sqrt(2/psi_normal - 1); 
            a = m_normal/(1+b_square);
            gaussian_noise = norm.ppf(uniform_sampling[i-1]);
            noise = gaussian_noise[psi<=1.5];
            V_normal = a * (np.sqrt(b_square) + noise)**2;
           
            V[i][psi<=1.5] = V_normal;
           
            psi_u = psi[psi>1.5];
            m_u = m[psi>1.5];
            # s_u = s_square[psi>1.5];
           
            p = (psi_u-1)/(psi_u+1);
            beta = (1-p)/m_u;
           
            u = uniform_sampling[i-1];
            unif_noise = u[psi>1.5];
           
            V_unif = 1/beta * np.log((1-p)/(1-unif_noise));
            V_unif[V_unif<0] = 0;
           
            V[i][psi_u>1.5] = V_unif;
            
            # generating the asset at the i-th step
            K0 = -dt * (rho * kappa * theta)/eta; 
            K1 = self.gamma1 * dt * (kappa*rho/eta -0.5) - rho/eta;
            K2 = self.gamma2 * dt * (kappa*rho/eta -0.5) + rho/eta;
            K3 = self.gamma1 * dt * (1 - rho**2);
            K4 = self.gamma2 * dt * (1 - rho**2);
            
            S[i] = S[i-1] * np.exp(mu*dt + K0 + K1*V[i-1] + K2*V[i] +np.sqrt(K3*V[i-1] + K4*V[i]) * norm.ppf(X_noise[i-1]));
            
        return (S, V);
    
    
    def characteristic_fun(self, u, X0, T, t=0):
        # this method computes the characteristic function of a stochastic process

        S0 = X0[0]
        nu0 = X0[1]
    
        mu = self.mu
        kappa = self.kappa
        theta = self.theta
        eta = self.eta
        rho = self.rho
    
        tau = T - t
    
        d = np.sqrt((kappa - 1j * rho * eta * u)**2 + eta**2 * (u**2 + 1j*u))
    
        g = ( kappa - 1j*rho*eta*u - d) / (kappa - 1j*rho*eta*u + d)
    
        C = (1j*u*(np.log(S0) + mu*tau) + (kappa*theta/eta**2)* ( (kappa - 1j*rho*eta*u - d)*tau - 2*np.log((1-g*np.exp(-d*tau))/(1-g) )))
    
        D = ((kappa - 1j*rho*eta*u - d) / eta**2* (1 - np.exp(-d*tau)) / ( 1 - g*np.exp(-d*tau) ))
    
        return np.exp(C + D*nu0)
    

    def cos_cumulants(self, X0, T):
        """
        Returns (c1, c2, c4)
        """
        S0 = X0[0]
        nu0 = X0[1]
    
        kappa = self.kappa
        theta = self.theta
        eta = self.eta
        rho = self.rho
        mu = self.mu
    
        expkt = np.exp(-kappa*T)
    
        c1 = (np.log(S0) + mu*T - 0.5*theta*T - (nu0-theta)*(1-expkt)/(2*kappa))
    
        c2 = (theta*T*eta**2/(8*kappa) * (4*rho*kappa/eta - 4*rho**2 + 2*kappa*T + 4*rho*kappa*T/eta - rho**2*kappa*T) + (nu0-theta) * eta**2 / (4*kappa**3) * (2*kappa*T*expkt - 3 + 4*expkt - np.exp(-2*kappa*T)) + nu0*(1-expkt)/kappa )
        
        c4 = 0.0
    
        return c1, c2, c4
        

class VarianceGamma(process):

    def __init__(self, mu, sigma, nu, theta):
        
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.theta = theta
        
    def set_parameters(self, parameters : np.array) -> None :
        
        self.mu = parameters[0]
        self.sigma = parameters[1]
        self.nu = parameters[2]
        self.theta = parameters[3]

    def simulate(self, X0, T, n_steps, n=10):

        S0 = X0[0]

        N_paths = 1 << n
        dt = T/n_steps

        S = np.empty((n_steps+1, N_paths))

        S[0] = S0

        # VG drift correction
        omega = (1/self.nu *np.log(1 - self.theta*self.nu -0.5*self.sigma**2*self.nu))

        for i in range(1,n_steps+1):

            # Gamma subordinator
            gamma = np.random.gamma(dt/self.nu, self.nu,N_paths)

            Z = np.random.normal(0, 1, N_paths)

            dX = (self.mu*dt + self.theta*gamma + self.sigma*np.sqrt(gamma)*Z + omega*dt)

            S[i] = S[i-1]*np.exp(dX)

        return S



    def characteristic_fun(self, u, X0, T, t=0):

        S0 = X0[0]

        tau = T-t 
        
        omega = (1/self.nu * np.log(1 - self.theta*self.nu -0.5*self.sigma**2*self.nu))

        # log-price characteristic function

        exponent = ( 1j*u*(np.log(S0)+self.mu*tau+omega*tau))

        VG = (1-1j * self.theta*self.nu*u + 0.5*self.sigma**2*self.nu*u**2)

        return np.exp(exponent)*VG**(-tau/self.nu)



    def cos_cumulants(self,X0,T):

        S0=X0[0]

        omega = (1/self.nu * np.log(1 -self.theta*self.nu - 0.5*self.sigma**2*self.nu))

        # cumulants of VG log process

        c1 = (np.log(S0) + (self.mu+omega+self.theta)*T)

        c2 = (T*(self.sigma**2 + self.theta**2*self.nu))

        # COS only requires c4 approximation
        c4 = (3*T * (self.sigma**4*self.nu + 2*self.theta**2*self.sigma**2*self.nu**2 + self.theta**4*self.nu**3))

        return c1,c2,c4
    
    
##########################################
############# MERTON MODEL ###############
##########################################


class Merton(process):

    def __init__(self, mu=np.nan, sigma=np.nan, lambda_jump=np.nan, mu_jump=0.0, sigma_jump=np.nan):
        """
        Merton Jump-Diffusion Model

        dS_t / S_t = mu dt + sigma dW_t + (J - 1)dN_t

        where

        N_t ~ Poisson(lambda_jump * t)

        log(J) ~ N(mu_jump, sigma_jump^2)

        Parameters
        ----------
        mu : float
            Drift of the asset.

        sigma : float
            Diffusion volatility.

        lambda_jump : float
            Jump intensity.

        mu_jump : float
            Mean of the logarithmic jump size.

        sigma_jump : float
            Volatility of the logarithmic jump size.
        """

        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump


    def set_parameters(self, parameters: np.array) -> None:
        """
        Set model parameters.

        parameters =
        [mu, sigma, lambda_jump, mu_jump, sigma_jump]
        """

        self.mu = parameters[0]
        self.sigma = parameters[1]
        self.lambda_jump = parameters[2]
        self.mu_jump = parameters[3]
        self.sigma_jump = parameters[4]


    def simulate(self, X0, T, n_steps, n=10):
        """
        Monte Carlo simulation of the Merton model.

        Parameters
        ----------
        X0 : Sequence[float]
            Initial state. X0[0] = S0.

        T : float
            Maturity in years.

        n_steps : int
            Number of time steps.

        n : int
            Number of simulations is N = 2^n.

        Returns
        -------
        S : np.ndarray
            Array of shape (n_steps + 1, N).
        """

        S0 = X0[0]

        # Number of Monte Carlo paths
        N = 1 << n

        # Time step
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Allocate paths
        S = np.empty((n_steps + 1, N))

        # Initial value
        S[0] = S0

        # ---------------------------------------------------------
        # Jump compensator
        #
        # E[J - 1]
        #     = E[exp(Y) - 1]
        #
        # where
        #
        # Y ~ N(mu_jump, sigma_jump^2)
        #
        # E[J] = exp(mu_jump + 0.5*sigma_jump^2)
        # ---------------------------------------------------------

        kappa_J = (np.exp(self.mu_jump + 0.5 * self.sigma_jump**2)- 1.0)

        jump_compensator = (-self.lambda_jump * kappa_J * dt)


        for i in range(1, n_steps + 1):

            # Brownian motion (antithetic variate)
            if N == 1:

                Z = np.random.normal(0.0, 1.0)

            else:

                Z = np.random.normal(0.0, 1.0, N // 2)
                Z = np.hstack((Z, -Z))

            # Diffusion component
            diffusion = ((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * sqrt_dt * Z)

            # Generate number of jumps

            n_jumps = np.random.poisson(self.lambda_jump * dt, N)

            # jump component
            jump_Z = np.random.normal(0.0, 1.0, N)

            jump_component = (n_jumps * self.mu_jump + self.sigma_jump* np.sqrt(n_jumps) * jump_Z)
            
            # Update the asset
            S[i] = S[i - 1] * np.exp(diffusion + jump_compensator + jump_component)

        return S


    def characteristic_fun(self, u, X0, T, t=0):
        """
        Characteristic function of log(S_T).

        phi(u) = E[exp(i u log(S_T))]
        """

        S0 = X0[0]

        tau = T - t

        # Jump compensator
        kappa_J = (np.exp(self.mu_jump  + 0.5 * self.sigma_jump**2) - 1.0)

        # Characteristic function of log-price
        drift = (np.log(S0) + (self.mu - 0.5 * self.sigma**2 - self.lambda_jump * kappa_J) * tau)

        diffusion = (-0.5 * self.sigma**2 * u**2 * tau)

        # Characteristic function of one log-jump
        phi_jump = np.exp(1j * u * self.mu_jump - 0.5 * self.sigma_jump**2 * u**2)

        # Compound Poisson contribution
        jump = (self.lambda_jump * tau * (phi_jump - 1.0))

        exponent = (1j * u * drift + diffusion + jump)

        return np.exp(exponent)


    def cos_cumulants(self, X0, T):
        """
        Cumulants of log(S_T).

        Returns
        -------
        c1 : first cumulant
        c2 : second cumulant
        c4 : fourth cumulant
        """

        S0 = X0[0]

        # Jump expectation
        kappa_J = (np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1.0)

        c1 = (np.log(S0) + (self.mu - 0.5 * self.sigma**2 - self.lambda_jump * kappa_J) * T + self.lambda_jump * T * self.mu_jump)

        c2 = (self.sigma**2 + self.lambda_jump * (self.mu_jump**2 + self.sigma_jump**2)) * T

        c4 = (self.lambda_jump * (self.mu_jump**4 + 6.0 * self.mu_jump**2 * self.sigma_jump**2 + 3.0 * self.sigma_jump**4) * T)

        return c1, c2, c4
    

###################################
########## BATES MODEL ############
###################################


class Bates(process):

    def __init__(self, mu=np.nan, kappa=np.nan, theta=np.nan, eta=np.nan, rho=np.nan, lam=np.nan, muj=np.nan, sigmaj=np.nan):

        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.rho = rho

        self.lam = lam
        self.muj = muj
        self.sigmaj = sigmaj
        
    def set_parameters(self, parameters : np.array) -> None :
        
        self.mu = parameters[0]
        self.kappa = parameters[1] 
        self.theta = parameters[2]
        self.eta = parameters[3] 
        self.rho = parameters[4] 
        self.lam = parameters[5]
        self.muj = parameters[6]
        self.sigmaj = parameters[7]
        

    ####################################################################
    # Monte Carlo simulation
    ####################################################################

    def simulate(self, X0, T, n_steps, n=10):

        S0 = X0[0]
        v0 = X0[1]

        N = 1 << n
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        S = np.empty((n_steps + 1, N))
        V = np.empty((n_steps + 1, N))

        S[0] = S0
        V[0] = v0

        # jump compensator        
        jump_laplace_transform = np.exp(self.muj + .5*(self.sigmaj**2));
        
        jump_compensator = self.lam * dt * (1 - jump_laplace_transform);

        for i in range(1, n_steps + 1):
            
            ## Geerating brownian_motions 
            
            if N==1: 
                
                Z1 = np.random.normal(0,1)
                Z2 = np.random.normal(0,1) 
                
                dBM1 = sqrt_dt * Z1 
                dBM2 = sqrt_dt * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
            
            else:
                
                Z1 = np.random.normal(0,1, N//2)
                Z1 = np.hstack((Z1, -Z1))
                
                Z2 = np.random.normal(0,1, N//2) 
                Z2 = np.hstack((Z2, -Z2))
                
                dBM1 = sqrt_dt * Z1 
                dBM2 = sqrt_dt * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) 
            
            ### vol part 
            
            V_pred = V[i-1] 
            v = V_pred + self.kappa * (self.theta - V_pred) * dt + self.eta * np.sqrt(V_pred) * dBM1 
            v[v<0] = 0
            V[i] = v
            
            #### jumps 
            
            n_jumps = np.random.poisson(self.lam*dt, N) 
            
            Jumps = self.muj * n_jumps + self.sigmaj*np.sqrt(n_jumps) * np.random.normal(0,1,N) 
            
            ### stock part 
                        
            S[i] = S[i-1] * np.exp((self.mu - .5*V[i-1]) * dt +  np.sqrt(V[i-1]) * dBM2 + jump_compensator + Jumps)

        return S, V


    ####################################################################
    # Characteristic function
    ####################################################################

    def characteristic_fun(self, u, X0, T, t=0):

        S0 = X0[0]
        v0 = X0[1]

        tau = T-t
        
        # Heston93 part of the characteristic function

        d = np.sqrt((self.kappa-1j*self.rho*self.eta*u)**2 + self.eta**2*(u**2+1j*u))

        g = (self.kappa - 1j*self.rho*self.eta*u - d) / (self.kappa - 1j*self.rho*self.eta*u + d)

        C = (1j*u*np.log(S0) + 1j*u*self.mu*tau + (self.kappa*self.theta/self.eta**2) * ((self.kappa -1j*self.rho*self.eta*u - d)*tau - 2*np.log( (1-g*np.exp(-d*tau))/ (1-g))))

        D = ((self.kappa - 1j*self.rho*self.eta*u - d) / self.eta**2 * (1-np.exp(-d*tau)) / (1-g*np.exp(-d*tau)))

        # Jump characteristic function (Compund Poisson processs)

        phiJ = np.exp(1j*u*self.muj - 0.5*self.sigmaj**2*u**2)

        kJ = np.exp( self.muj + 0.5*self.sigmaj**2) - 1.0

        jump_part = self.lam*tau*(phiJ - 1 - 1j*u*kJ)

        return np.exp(C + D*v0 + jump_part)


    ####################################################################
    # COS cumulants
    ####################################################################

    def cos_cumulants(self, X0, T):

        S0 = X0[0]
        v0 = X0[1]

        expkt = np.exp(-self.kappa*T)

        c1 = (np.log(S0) + (self.mu - self.lam*(np.exp(self.muj+0.5*self.sigmaj**2)-1))*T - 0.5*self.theta*T - (v0-self.theta)*(1-expkt)/(2*self.kappa))

        c2 = (self.theta*T*self.eta**2/(8*self.kappa) * (4*self.rho*self.kappa/self.eta - 4*self.rho**2 + 2*self.kappa*T + 4*self.rho*self.kappa*T/self.eta - self.rho**2*self.kappa*T) + (v0-self.theta) * self.eta**2 / (4*self.kappa**3) * (2*self.kappa*T*expkt-3 + 4*expkt - np.exp(-2*self.kappa*T)) + v0*(1-expkt)/self.kappa + self.lam*T*(self.sigmaj**2+self.muj**2))

        c4 = (3*self.lam*T *( self.sigmaj**4 + 6*self.muj**2*self.sigmaj**2 + self.muj**4))

        return c1, c2, c4