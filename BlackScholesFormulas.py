# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:07:11 2026

@author: Nicola
"""

import numpy as np
from scipy.stats import norm


# Black-Scholes auxiliary functions

def d1(S0, K, T, r, q, sigma):
    """
    Black-Scholes d1.
    """
    return (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, T, r, q, sigma):
    """
    Black-Scholes d2.
    """
    return d1(S0, K, T, r, q, sigma) - sigma * np.sqrt(T)


# European Call

def EuroCall_BS(S0, K, T, r, sigma, q=0.0):

    D1 = d1(S0, K, T, r, q, sigma)
    D2 = d2(S0, K, T, r, q, sigma)

    return (S0 * np.exp(-q * T) * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2))


# ============================================================
# European Put
# ============================================================

def EuroPut_BS(S0, K, T, r, sigma, q=0.0):

    D1 = d1(S0, K, T, r, q, sigma)
    D2 = d2(S0, K, T, r, q, sigma)

    return (K * np.exp(-r * T) * norm.cdf(-D2) - S0 * np.exp(-q * T) * norm.cdf(-D1))


# European Call Digital (Cash-or-Nothing)

def EuroCallDigital_BS(S0, K, T, r, sigma, q=0.0, payout=1.0):

    D2 = d2(S0, K, T, r, q, sigma)

    return payout * np.exp(-r * T) * norm.cdf(D2)


# European Put Digital (Cash-or-Nothing)

def EuroPutDigital_BS(S0, K, T, r, sigma, q=0.0, payout=1.0):

    D2 = d2(S0, K, T, r, q, sigma)

    return payout * np.exp(-r * T) * norm.cdf(-D2)


# European Call Asset-or-Nothing

def EuroCallAssetOrNothing_BS(S0, K, T, r, sigma, q=0.0):

    D1 = d1(S0, K, T, r, q, sigma)

    return S0 * np.exp(-q * T) * norm.cdf(D1)


# European Put Asset-or-Nothing

def EuroPutAssetOrNothing_BS(S0, K, T, r, sigma, q=0.0):

    D1 = d1(S0, K, T, r, q, sigma)

    return S0 * np.exp(-q * T) * norm.cdf(-D1) 


########### Asian options


def geometric_asian_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Kemna-Vorst closed-form price for a continuously
    sampled geometric-average Asian call option.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility of the underlying.
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        Asian call price.
    """

    if S0 <= 0:
        raise ValueError("S0 must be positive.")

    if K <= 0:
        raise ValueError("K must be positive.")

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if T <= 0:
        return max(S0 - K, 0.0)

    # Effective volatility of the geometric average
    sigma_g = sigma / np.sqrt(3.0)

    # Forward price of the geometric average
    F_g = S0 * np.exp((r - sigma**2 / 6.0) * T)

    # Standard Black-Scholes quantities
    D1 = (np.log(F_g / K) + 0.5 * sigma_g**2 * T) / (sigma_g * np.sqrt(T))
    D2 = D1 - sigma_g * np.sqrt(T)

    # Discounted Black-Scholes price
    price = np.exp(-r * T) * (F_g * norm.cdf(D1) - K * norm.cdf(D2))

    return price



def geometric_asian_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Kemna-Vorst closed-form price for a continuously
    sampled geometric-average Asian put option.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility of the underlying.
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        Asian put price.
    """

    if S0 <= 0:
        raise ValueError("S0 must be positive.")

    if K <= 0:
        raise ValueError("K must be positive.")

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if T <= 0:
        return max(K - S0, 0.0)

    # Effective volatility of the geometric average
    sigma_g = sigma / np.sqrt(3.0)

    # Forward price of the geometric average
    F_g = S0 * np.exp((r - sigma**2 / 6.0) * T)

    # Standard Black-Scholes quantities
    D1 = (np.log(F_g / K) + 0.5 * sigma_g**2 * T) / (sigma_g * np.sqrt(T))

    D2 = D1 - sigma_g * np.sqrt(T)

    # Discounted Black-Scholes put price
    price = np.exp(-r * T) * ( K * norm.cdf(-D2) - F_g * norm.cdf(-D1))

    return price