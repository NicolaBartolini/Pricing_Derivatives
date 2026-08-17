# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:38:10 2026

@author: Nicola
""" 
import os 
import sys 

root = os.getcwd().split("\\")[:-1]
sys.path.append(os.path.join('\\'.join(root)))

import pytest
import numpy as np
from datetime import datetime, timedelta

from process_class import VarianceGamma


# ----------------------------------------------------
# Variance Gamma martingale test
# ----------------------------------------------------

PARAMETERS = [
    # mu, sigma, nu, theta
    (0.05, 0.20, 0.20, -0.10),
    (0.03, 0.15, 0.30, -0.20),
    (0.07, 0.25, 0.10, 0.05),
    (0.05, 0.30, 0.15, 0.10),
    (0.02, 0.10, 0.40, -0.05)]


def test_variance_gamma_martingale():


    S0 = 1.0
    r = 0.05
    T = 1.0
    n_steps = 252
    np.random.seed(42)

    for mu,sigma,nu,theta in PARAMETERS:

        process = VarianceGamma(mu=mu, sigma=sigma, nu=nu, theta=theta)

        ST = process.simulate(X0=[S0], T=T, n_steps=n_steps, n=20)[-1]

        discounted_mean = (np.mean(ST) * np.exp(-r*T))

        expected = S0

        error = discounted_mean - expected

        print("\nParameters:", mu, sigma, nu, theta)

        print("Discounted E[ST] =", discounted_mean)

        print("S0 =", expected)

        print("Error =", error)

        np.testing.assert_allclose(discounted_mean, expected, rtol=2e-2, atol=2e-1)