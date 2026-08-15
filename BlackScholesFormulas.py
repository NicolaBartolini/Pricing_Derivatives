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