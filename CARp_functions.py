# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:20:31 2026

@author: Nicola
"""

import numpy as np
import scipy.linalg as la
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar 

##############################################
########## Simulate CAR(p) process############ 
##############################################

def simulate_car_process(p, alpha, T, dt=1.0, sigma_fn=None, initial_state=None, seed=42):
    """
    Simulates a Continuous Autoregressive CAR(p) process for p = 1, 2, or 3.
    
    Parameters:
    -----------
    p : int
        Order of the CAR process (1, 2, or 3).
    alpha : list or np.ndarray
        Array of positive parameters [alpha_1, ..., alpha_p] of length p.
    T : int
        Total number of discrete time steps to simulate.
    dt : float, optional
        Time step size (default is 1.0 for daily resolution).
    sigma_fn : function, optional
        A function sigma(t) returning the volatility at time t. 
        If None, a constant volatility of 1.0 is used.
    initial_state : np.ndarray, optional
        Initial vector state X(0) of shape (p,). If None, initialized to zeros.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    X_1 : np.ndarray
        Simulated primary process values X_1(t) over time.
    X_state : np.ndarray
        Full state vector evolution of shape (T, p).
    """
    if p not in [1, 2, 3]:
        raise ValueError("Order p must be 1, 2, or 3.")
    if len(alpha) != p:
        raise ValueError(f"For CAR({p}), alpha must contain exactly {p} parameters.")
    
    np.random.seed(seed)
    
    # Default volatility function (constant)
    if sigma_fn is None:
        sigma_fn = lambda t: 1.0
        
    # State vector initialization X(t) in R^p
    if initial_state is None:
        X = np.zeros(p)
    else:
        X = np.array(initial_state, dtype=float)
        
    # Construct the continuous companion matrix A
    A = np.zeros((p, p))
    if p > 1:
        for i in range(p - 1):
            A[i, i + 1] = 1.0
    A[-1, :] = -np.array(alpha)[::-1]  # Matrix row: [-alpha_p, ..., -alpha_1]
    
    # Vector unit selector e_p = (0, ..., 0, 1)^T
    e_p = np.zeros(p)
    e_p[-1] = 1.0
    
    # Storage arrays
    X_state = np.zeros((T, p))
    X_state[0] = X
    
    # Simulation Loop
    for t in range(1, T):
        sigma_t = sigma_fn(t)
        
        # Standard Normal noise increment standard scaled by sqrt(dt)
        epsilon_t = np.random.normal(0, 1)
        dB_t = np.sqrt(dt) * epsilon_t
        
        # Euler-Maruyama State Update: dX(t) = A * X(t) * dt + e_p * sigma_t * dB_t
        dX = np.dot(A, X) * dt + e_p * sigma_t * dB_t
        X = X + dX
        
        X_state[t] = X
        
    # The primary observable process is the first component X_1(t)
    X_1 = X_state[:, 0]
    return X_1, X_state

# =====================================================================
# 1. CAR(p) FITTING FUNCTION FROM TIME SERIES
# =====================================================================

def fit_car_model(series, p):
    """
    Fits a CAR(p) model to a discrete time series by:
    1. Estimating AR(p) coefficients via OLS.
    2. Mapping AR(p) parameters to continuous-time CAR(p) alphas.
    
    Parameters:
    -----------
    series : np.ndarray
        1D array of observed time series data (assumed mean-zero stationary).
    p : int
        CAR process order (1, 2, or 3).
        
    Returns:
    --------
    alpha_hat : np.ndarray
        Estimated continuous-time parameters [alpha_1, ..., alpha_p].
    phi_hat : np.ndarray
        Estimated discrete-time AR(p) coefficients.
    is_stable : bool
        True if all eigenvalues of matrix A have negative real parts.
    """
    N = len(series)
    if N <= p:
        raise ValueError("Series length must be greater than model order p.")
        
    # Construct OLS Design Matrix Y = X * phi + eps
    Y = series[p:]
    X_mat = np.column_stack([series[p - 1 - i : N - 1 - i] for i in range(p)])
    
    # Solve linear least squares: phi_hat = (X^T X)^(-1) X^T Y
    phi_hat, _, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
    
    # Map AR(p) -> CAR(p) parameters alpha
    alpha_hat = np.zeros(p)
    
    if p == 1:
        phi1 = phi_hat[0]
        alpha_hat[0] = 1.0 - phi1
        
    elif p == 2:
        phi1, phi2 = phi_hat[0], phi_hat[1]
        a1 = 2.0 - phi1
        a2 = a1 - phi2 - 1.0
        alpha_hat = np.array([a1, a2])
        
    elif p == 3:
        phi1, phi2, phi3 = phi_hat[0], phi_hat[1], phi_hat[2]
        a1 = 3.0 - phi1
        a2 = 2.0 * a1 - phi2 - 3.0
        a3 = 1.0 - a1 + a2 - phi3
        alpha_hat = np.array([a1, a2, a3])

    # Check stationarity/stability via Companion Matrix eigenvalues
    A = np.zeros((p, p))
    if p > 1:
        for i in range(p - 1):
            A[i, i + 1] = 1.0
    A[-1, :] = -alpha_hat[::-1]
    
    eigenvalues = np.linalg.eigvals(A)
    is_stable = np.all(np.real(eigenvalues) < 0)
    
    return alpha_hat, phi_hat, is_stable


# =====================================================================
# 2. EXACT ANALYTICAL PRICING FOR DIGITAL OPTION ON SPOT T_T
# =====================================================================

def car_companion_matrix(p, alpha):
    """Constructs the continuous companion matrix A for a CAR(p) process."""
    A = np.zeros((p, p))
    if p > 1:
        for i in range(p - 1):
            A[i, i + 1] = 1.0
    A[-1, :] = -np.array(alpha)[::-1]
    return A

# def exact_digital_call_price(p, alpha, t, T_mat, X_t, K, Lambda_T, r=0.03, 
#                              sigma_fn=None, theta_fn=None):
#     """
#     Computes the exact closed-form price of a Digital Call Option paying $1 
#     at maturity T_mat if T_{T_mat} >= K.
#     """
#     if sigma_fn is None:
#         sigma_fn = lambda u: 1.0
#     if theta_fn is None:
#         theta_fn = lambda u: 0.1  # Constant market price of risk
        
#     A = car_companion_matrix(p, alpha)
#     e1 = np.zeros(p); e1[0] = 1.0
#     ep = np.zeros(p); ep[-1] = 1.0
    
#     # 1. Initial condition deterministic contribution
#     term_init = e1 @ la.expm(A * (T_mat - t)) @ X_t
    
#     # 2. Market price of risk drift integral component
#     def drift_integrand(u):
#         exp_matrix = la.expm(A * (T_mat - u))
#         return (e1 @ exp_matrix @ ep) * sigma_fn(u) * theta_fn(u)
    
#     drift_int, _ = quad(drift_integrand, t, T_mat)
    
#     # Total conditional mean of X_1(T_mat)
#     mu_t_T = term_init + drift_int
    
#     # 3. Conditional variance integral component
#     def var_integrand(u):
#         exp_matrix = la.expm(A * (T_mat - u))
#         kernel = e1 @ exp_matrix @ ep
#         return (sigma_fn(u) * kernel) ** 2
    
#     var_int, _ = quad(var_integrand, t, T_mat)
#     v_t_T = np.sqrt(var_int)
    
#     # Total mean of spot temperature T_{T_mat}
#     mean_T = Lambda_T + mu_t_T
    
#     # Black-Scholes type Gaussian probability d_2 term
#     d2 = (mean_T - K) / v_t_T
    
#     # Discounted risk-neutral expectation
#     price = np.exp(-r * (T_mat - t)) * norm.cdf(d2)
    
#     return price, mean_T, v_t_T 



def exact_digital_call_price(theta_const, p, alpha, t, T_mat, X_t, K, Lambda_T, r=0.03, sigma_fn=None):
    """
    Computes the exact closed-form price of a Digital Call Option with constant theta.
    
    Parameters:
    -----------
    theta_const : float
        The constant market price of risk to be calibrated.
    p : int
        Order of the CAR process (1, 2, or 3).
    alpha : list or np.ndarray
        CAR(p) model coefficients.
    t : float
        Valuation time.
    T_mat : float
        Option maturity time.
    X_t : np.ndarray
        Current state vector X(t) of shape (p,).
    K : float
        Option strike.
    Lambda_T : float
        Deterministic trend/seasonality at maturity T_mat.
    r : float
        Risk-free interest rate.
    sigma_fn : function, optional
        Seasonal volatility function sigma(u). Defaults to constant 1.0.
    """
    if sigma_fn is None:
        sigma_fn = lambda u: 1.0
        
    A = car_companion_matrix(p, alpha)
    e1 = np.zeros(p); e1[0] = 1.0
    ep = np.zeros(p); ep[-1] = 1.0
    
    # 1. Initial state contribution
    term_init = e1 @ la.expm(A * (T_mat - t)) @ X_t
    
    # 2. Risk-neutral drift integral (factoring out constant theta)
    def drift_kernel(u):
        exp_matrix = la.expm(A * (T_mat - u))
        try:
            res = (e1 @ exp_matrix @ ep) * sigma_fn(u)
        except:
            res = (e1 @ exp_matrix @ ep) * sigma_fn 
        return res
    
    drift_int, _ = quad(drift_kernel, t, T_mat)
    mu_t_T = term_init + theta_const * drift_int
    
    # 3. Conditional variance integral
    def var_kernel(u):
        exp_matrix = la.expm(A * (T_mat - u))
        
        try:
            res = (sigma_fn(u) * (e1 @ exp_matrix @ ep)) ** 2
        except:
            res = (sigma_fn * (e1 @ exp_matrix @ ep)) ** 2 
        
        # if sigma_fn is None:
        #     sigma_fn = lambda u: 1.0
        # elif not callable(sigma_fn):
        #     val = float(sigma_fn)
        #     sigma_fn = lambda u: val
                
        return res
    
    var_int, _ = quad(var_kernel, t, T_mat)
    v_t_T = np.sqrt(var_int)
    
    # Option pricing via standard normal CDF
    mean_T = Lambda_T + mu_t_T
    d2 = (mean_T - K) / v_t_T
    
    return np.exp(-r * (T_mat - t)) * norm.cdf(d2)

# =====================================================================
# 3. OBJECTIVE FUNCTION & CALIBRATION ROUTINE
# =====================================================================

def calibrate_constant_theta(market_options, p, alpha, X_t, r=0.03, sigma_fn=None, bounds=(-2.0, 2.0)):
    """
    Calibrates constant market price of risk (theta) by minimizing MSE against market prices.
    
    Parameters:
    -----------
    market_options : list of dicts
        List containing market contract information. Each dict must specify:
        {'t': float, 'T_mat': float, 'K': float, 'Lambda_T': float, 'market_price': float}
    p : int
        CAR process order (1, 2, or 3).
    alpha : list or np.ndarray
        Analyst-provided CAR parameters.
    X_t : np.ndarray
        Analyst-provided current state vector.
    r : float
        Risk-free rate.
    sigma_fn : function, optional
        Volatility function provided by analyst.
    bounds : tuple
        Optimization search bounds for constant theta.
        
    Returns:
    --------
    calibrated_theta : float
        The calibrated constant theta.
    min_mse : float
        The minimum Mean Squared Error achieved.
    """
    
    def mse_objective(theta):
        errors = []
        for contract in market_options:
            model_price = exact_digital_call_price(
                theta_const=theta,
                p=p,
                alpha=alpha,
                t=contract['t'],
                T_mat=contract['T_mat'],
                X_t=X_t,
                K=contract['K'],
                Lambda_T=contract['Lambda_T'],
                r=r,
                sigma_fn=sigma_fn
            )
            err = model_price - contract['market_price']
            errors.append(err ** 2)
            
        return np.mean(errors)

    # Perform 1D scalar minimization over specified bounds
    result = minimize_scalar(mse_objective, bounds=bounds, method='bounded')
    
    if not result.success:
        raise RuntimeError(f"Calibration failed: {result.message}")
        
    return result.x, result.fun 


# =====================================================================
# 4. EXAMPLE USAGE & VERIFICATION
# =====================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Analyst-Provided Fixed Parameters
    # -----------------------------------------------------------------
    p = 3
    alpha = [2.09, 1.38, 0.22]              # Fitted CAR(3) parameters
    X_t = np.array([1.2, 0.1, -0.05])        # Current state vector
    r = 0.03                                 # Interest rate
    # sigma_fn = lambda u: 1.5 + 0.5 * np.cos(2 * np.pi * u) # Volatility function
    sigma_fn = 1.5
    # -----------------------------------------------------------------
    # Simulated Market Data Across Multiple Contracts (Strikes & Maturities)
    # (True synthetic theta used to generate mock prices: theta_true = -0.125)
    # -----------------------------------------------------------------
   
    true_theta = -0.125
    # true_theta = 0
    
    contracts_metadata = [
        {'t': 0.0, 'T_mat': 0.5, 'K': 17.0, 'Lambda_T': 16.0},
        {'t': 0.0, 'T_mat': 0.5, 'K': 18.0, 'Lambda_T': 16.0},
        {'t': 0.0, 'T_mat': 1.0, 'K': 17.5, 'Lambda_T': 16.5},
        {'t': 0.0, 'T_mat': 1.0, 'K': 18.5, 'Lambda_T': 16.5},
        {'t': 0.0, 'T_mat': 1.5, 'K': 19.0, 'Lambda_T': 17.0},
    ]
    
    # Generate market prices (adding small Gaussian noise to simulate real data)
    np.random.seed(42)
    market_options = []
    for c in contracts_metadata:
        clean_price = exact_digital_call_price(
            theta_const=true_theta, p=p, alpha=alpha, t=c['t'], 
            T_mat=c['T_mat'], X_t=X_t, K=c['K'], Lambda_T=c['Lambda_T'], 
            r=r, sigma_fn=sigma_fn
        )
        noisy_price = max(0.0, clean_price + np.random.normal(0, 0.002))
        market_options.append({**c, 'market_price': noisy_price})

    # -----------------------------------------------------------------
    # Perform Calibration
    # -----------------------------------------------------------------
    calibrated_theta, min_mse = calibrate_constant_theta(
        market_options=market_options,
        p=p,
        alpha=alpha,
        X_t=X_t,
        r=r,
        sigma_fn=sigma_fn,
        bounds=(-1.0, 1.0)
    )

    # -----------------------------------------------------------------
    # Print Results & Comparison Table
    # -----------------------------------------------------------------
    print("=" * 80)
    print(f"CALIBRATION RESULTS (CAR({p}) Model)")
    print("=" * 80)
    print(f"True Synthetic Theta : {true_theta:.6f}")
    print(f"Calibrated Theta     : {calibrated_theta:.6f}")
    print(f"Minimized MSE        : {min_mse:.8e}")
    print("=" * 80)
    
    print(f"{'Option #':<8} | {'Maturity':<10} | {'Strike':<8} | {'Market Price':<14} | {'Model Price':<14} | {'Abs Error':<10}")
    print("-" * 80)
    
    for idx, c in enumerate(market_options, 1):
        m_price = c['market_price']
        mod_price = exact_digital_call_price(
            theta_const=calibrated_theta, p=p, alpha=alpha, t=c['t'], 
            T_mat=c['T_mat'], X_t=X_t, K=c['K'], Lambda_T=c['Lambda_T'], 
            r=r, sigma_fn=sigma_fn
        )
        err = abs(m_price - mod_price)
        print(f"{idx:<8} | {c['T_mat']:<10.2f} | {c['K']:<8.1f} | {m_price:<14.6f} | {mod_price:<14.6f} | {err:<10.6f}")
        
    print("=" * 80)