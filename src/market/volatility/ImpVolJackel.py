# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:40:17 2026

@author: Nicola
"""

import numpy as np 
from scipy.stats import norm
from scipy.special import ndtr, ndtri

SQRT_2PI = np.sqrt(2.0 * np.pi)
SQRT_PI_OVER_6 = np.pi / 6.0
R_MIN = -1.0 + np.sqrt(np.finfo(float).eps)
R_MAX = 2.0 / np.finfo(float).eps**2


def normalized_black_call(x: float, s: float) -> float:
    """Normalized OTM Black call, with x <= 0 and s > 0."""
    if s <= 0.0:
        return 0.0

    d1 = x / s + 0.5 * s
    d2 = x / s - 0.5 * s

    return np.exp(0.5 * x) * norm.cdf(d1) - np.exp(-0.5 * x) * norm.cdf(d2)



def normalize_black_option(price: float, F: float, K: float, discount_factor: float, flag: str):
    """
    Convert any European call or put into an equivalent OTM call.

    Parameters
    ----------
    price : float
        Market option price (discounted).
    F : float
        Forward price.
    K : float
        Strike.
    discount_factor : float
        Discount factor D(0,T).
    flag : str
        'c' for call, 'p' for put.

    Returns
    -------
    beta : float
        Normalized OTM time value.
    x : float
        Negative log-moneyness.
    """

    if flag not in ("c", "p"):
        raise ValueError("flag must be 'c' or 'p'.")

    if F <= 0.0 or K <= 0.0 or discount_factor <= 0.0:
        raise ValueError("F, K and discount_factor must be positive.")

    # Remove discounting
    price_undiscounted = price / discount_factor

    # Convert to an OTM call
    if F <= K:

        # OTM call
        if flag == "c":
            otm_call_price = price_undiscounted
        else:
            # Put is ITM: P = C + K - F
            otm_call_price = price_undiscounted - (K - F)

        x = np.log(F / K)

    else:

        # Call is ITM: C = P + F - K
        if flag == "c":
            otm_call_price = price_undiscounted - (F - K)
        else:
            # OTM put -> equivalent OTM call under normalized symmetry
            otm_call_price = price_undiscounted

        x = -np.log(F / K)

    # Normalize
    beta = otm_call_price / np.sqrt(F * K)

    return beta, x


def normalized_vega(x: float, s: float) -> float:
    """First derivative of normalized Black price with respect to s."""
    if s <= 0.0:
        return 0.0

    exponent = -0.5 * ((x / s)**2 + (0.5 * s)**2)

    return np.exp(exponent) / SQRT_2PI


def normalized_black_derivatives(x: float, s: float):
    """
    Return b, b', b'', b'''.

    H2 = b'' / b'
    H3 = b''' / b'
    """
    b = normalized_black_call(x, s)
    bp = normalized_vega(x, s)

    if s <= 0.0:
        raise ValueError("s must be strictly positive.")

    H2 = x**2 / s**3 - s / 4.0
    H3 = H2**2 - 3.0 * x**2 / s**4 - 1.0 / 4.0

    bpp = bp * H2
    bppp = bp * H3

    return b, bp, bpp, bppp



def rational_cubic(z: float, zl: float, zr: float, yl: float, yr: float, dl: float, dr: float, r: float) -> float:
    """Delbourgo-Gregory rational cubic interpolation."""

    h = zr - zl

    if h <= 0.0:
        raise ValueError("Interpolation interval must be positive.")

    t = (z - zl) / h
    omt = 1.0 - t

    if r >= R_MAX:
        return yl * omt + yr * t

    numerator = (yr * t**3 + (r * yr - h * dr) * t**2 * omt + (r * yl + h * dl) * t * omt**2 + yl * omt**3)

    denominator = 1.0 + (r - 3.0) * t * omt

    return numerator / denominator


def minimum_rational_cubic_parameter(dl: float, dr: float, secant: float) -> float:
    """
    Lower bound on r used to preserve monotonicity/convexity.
    """
    monotonic = dl * secant >= 0.0 and dr * secant >= 0.0
    convex = dl <= secant <= dr
    concave = dl >= secant >= dr

    if not monotonic and not convex and not concave:
        return R_MIN

    r1 = -np.inf
    r2 = -np.inf

    if monotonic and secant != 0.0:
        r1 = (dr + dl) / secant

    if convex or concave:
        if secant != dl and dr != secant:
            
            difference = dr - dl
            
            r2 = max(abs(difference / (dr - secant)), abs(difference / (secant - dl)),)

    return max(R_MIN, r1, r2)


def rational_cubic_parameter_left(zl: float, zr: float, yl: float, yr: float, dl: float, dr: float, ypp_left: float) -> float:
    """Choose r to match the second derivative at the left endpoint."""

    h = zr - zl
    secant = (yr - yl) / h

    numerator = 0.5 * h * ypp_left + dr - dl
    denominator = secant - dl

    if abs(denominator) < np.finfo(float).tiny:
        r = R_MAX if numerator > 0.0 else R_MIN
    else:
        r = numerator / denominator

    r_min = minimum_rational_cubic_parameter(dl, dr, secant)

    return max(r, r_min)


def rational_cubic_parameter_right(zl: float, zr: float, yl: float, yr: float, dl: float, dr: float, ypp_right: float) -> float:
    """Choose r to match the second derivative at the right endpoint."""

    h = zr - zl
    secant = (yr - yl) / h

    numerator = 0.5 * h * ypp_right + dr - dl
    denominator = dr - secant

    if abs(denominator) < np.finfo(float).tiny:
        r = R_MAX if numerator > 0.0 else R_MIN
    else:
        r = numerator / denominator

    r_min = minimum_rational_cubic_parameter(dl, dr, secant)

    return max(r, r_min)


def lower_map(x: float, s: float):
    """Return f_l, df_l/dbeta and d2f_l/dbeta2 at (x, s)."""

    ax = abs(x)
    z = ax / (np.sqrt(3.0) * s)
    y = z**2
    s2 = s**2

    Phi = ndtr(-z)
    phi = np.exp(-0.5 * z**2) / SQRT_2PI

    f = 2.0 * np.pi / np.sqrt(27.0) * ax * Phi**3

    fp = 2.0 * np.pi * y * Phi**2 * np.exp(y + 0.125 * s2)

    fpp = SQRT_PI_OVER_6 * y / (s2 * s) * Phi * (8.0 * np.sqrt(3.0) * s * ax + (3.0 * s2 * (s2 - 8.0) - 8.0 * x**2) * Phi / phi) * np.exp(2.0 * y + 0.25 * s2)

    return f, fp, fpp


def inverse_lower_map(x: float, f: float) -> float:
    """Invert the lower transformation."""

    if f <= 0.0:
        return 0.0

    ax = abs(x)

    probability = (f / (2.0 * np.pi / np.sqrt(27.0) * ax)) ** (1.0 / 3.0)

    probability = np.clip(probability, np.finfo(float).tiny, 0.5 - np.finfo(float).eps,)

    return abs(ax / (np.sqrt(3.0) * ndtri(probability)))


def upper_map(s: float) -> float:
    """f_u(s) = Phi(-s/2)."""
    return ndtr(-0.5 * s)


def upper_map_derivatives(x: float, s: float):
    """
    Return f_u, df_u/dbeta and d2f_u/dbeta2.

    The derivatives are with respect to beta, using
    the inverse Black relationship.
    """

    f = upper_map(s)

    w = (x / s)**2

    fp = -0.5 * np.exp(0.5 * w)

    fpp = (np.sqrt(np.pi / 2.0) * np.exp(w + 0.125 * s**2) * w / s)

    return f, fp, fpp


def inverse_upper_map(f: float) -> float:
    """Invert f_u(s) = Phi(-s/2)."""

    f = np.clip(f,np.finfo(float).tiny, 0.5 - np.finfo(float).eps,)

    return -2.0 * ndtri(f) 


def initial_guess(beta: float, x: float):
    """
    Four-branch rational-cubic initial guess.

    Returns
    -------
    s0 : float
        Initial total volatility.
    branch : str
        lower, central_left, central_right or upper.
    """

    if x > 0.0:
        raise ValueError("This implementation expects x <= 0.")

    if beta <= 0.0:
        return 0.0, "zero"

    # Exact ATM inverse.
    if abs(x) < 1e-14:
        return 2.0 * ndtri(0.5 * (beta + 1.0)), "atm"

    bmax = np.exp(0.5 * x)

    if beta >= bmax:
        raise ValueError("Normalized price must be below bmax.")

    # Inflection point.
    sc = np.sqrt(-2.0 * x)
    bc, vc, _, _ = normalized_black_derivatives(x, sc)

    # Lower boundary.
    sl = max(sc - bc / vc, np.finfo(float).tiny)
    bl = normalized_black_call(x, sl)

    if beta < bl:

        # Lower transformed interpolation.
        fl, fpl, fppl = lower_map(x, sl)

        r = rational_cubic_parameter_right(0.0, bl, 0.0, fl, 1.0, fpl, fppl)

        f = rational_cubic(beta, 0.0, bl, 0.0, fl, 1.0, fpl, r)

        # Fallback if roundoff gives a non-positive value.
        if f <= 0.0:
            t = beta / bl
            f = (fl * t + bl * (1.0 - t)) * t

        s0 = inverse_lower_map(x, f)

        return s0, "lower"

    if beta <= bc:

        # Left central branch: [bl, bc].
        vl = normalized_vega(x, sl)
        dl = 1.0 / vl
        dc = 1.0 / vc

        r = rational_cubic_parameter_right(bl, bc, sl, sc, dl, dc, 0.0)

        s0 = rational_cubic(beta, bl, bc, sl, sc, dl, dc, r)

        return s0, "central_left"

    # Upper boundary.
    sh = sc + (bmax - bc) / vc
    bh = normalized_black_call(x, sh)

    if beta <= bh:

        # Right central branch: [bc, bh].
        vh = normalized_vega(x, sh)
        dc = 1.0 / vc
        dh = 1.0 / vh

        r = rational_cubic_parameter_left(bc, bh, sc, sh, dc, dh, 0.0)

        s0 = rational_cubic(beta, bc, bh, sc, sh, dc, dh, r)

        return s0, "central_right"

    # Upper transformed interpolation.
    fh, fph, fpph = upper_map_derivatives(x, sh)

    r = rational_cubic_parameter_left(bh, bmax, fh, 0.0, fph, -0.5, fpph)

    f = rational_cubic(beta, bh, bmax, fh, 0.0, fph, -0.5, r)

    # Quadratic fallback.
    if f <= 0.0:
        h = bmax - bh
        t = (beta - bh) / h
        f = (fh * (1.0 - t) + 0.5 * h * t) * (1.0 - t)

    s0 = inverse_upper_map(f)

    return s0, "upper"


def householder_terms(beta: float, x: float, s: float, branch: str):
    """
    Return Newton correction, g''/g' and g'''/g'.

    The expressions are algebraically equivalent to using
    g, g', g'' and g''' directly.
    """

    b, bp, _, _ = normalized_black_derivatives(x, s)

    if bp <= 0.0 or b <= 0.0:
        raise FloatingPointError("Black price or vega underflow.")

    H2 = x**2 / s**3 - s / 4.0
    H3 = H2**2 - 3.0 * x**2 / s**4 - 0.25

    if branch == "lower":

        log_b = np.log(b)
        log_beta = np.log(beta)
        bp_over_b = bp / b

        newton = (log_beta - log_b) * log_b / log_beta / bp_over_b

        halley = H2 - bp_over_b * (1.0 + 2.0 / log_b)

        hh3 = H3 + 2.0 * bp_over_b**2 * (1.0 + 3.0 / log_b * (1.0 + 1.0 / log_b)) - 3.0 * H2 * bp_over_b * (1.0 + 2.0 / log_b)

        return newton, halley, hh3

    bmax = np.exp(0.5 * x)

    if branch == "upper":

        distance = bmax - b

        if distance <= 0.0:
            raise FloatingPointError("Upper price limit reached.")

        g = np.log((bmax - beta) / distance)
        gp = bp / distance

        newton = -g / gp
        halley = H2 + gp
        hh3 = H3 + gp * (2.0 * gp + 3.0 * H2)

        return newton, halley, hh3

    # Central branches: g(s) = b(x,s) - beta.
    newton = (beta - b) / bp

    return newton, H2, H3 


def solve_normalized_implied_volatility(beta: float, x: float, max_iter: int = 20, tol: float = 1e-13):
    """Solve b(x, s) = beta using Jäckel's initial guess and Householder."""

    if beta <= 0.0:
        return 0.0

    bmax = np.exp(0.5 * x)

    if beta >= bmax:
        raise ValueError("No finite implied volatility for this price.")

    s, initial_branch = initial_guess(beta, x)

    if initial_branch == "atm":
        return s

    if initial_branch == "zero":
        return 0.0

    # The upper transformed objective is used only above bmax / 2.
    branch = initial_branch

    if branch == "upper" and beta <= 0.5 * bmax:
        branch = "central_right"

    s = max(s, 1e-12)

    # Establish a bracket.
    s_left = 0.0
    s_right = max(2.0 * s, 1e-6)

    while normalized_black_call(x, s_right) < beta:
        s_right *= 2.0

        if s_right > 1e6:
            raise RuntimeError("Could not bracket implied volatility.")

    # Ensure initial point is inside the bracket.
    if not (s_left < s < s_right):
        s = 0.5 * (s_left + s_right)

    eps = np.finfo(float).eps

    for _ in range(max_iter):

        b = normalized_black_call(x, s)

        # Update the bracket using monotonicity of b in s.
        if b > beta:
            s_right = s
        else:
            s_left = s

        # Check price convergence.
        if abs(b - beta) <= tol * max(beta, 1e-14):
            return s

        try:
            newton, halley, hh3 = householder_terms(beta, x, s, branch)

            denominator = (1.0 + newton * (halley + hh3 * newton / 6.0))

            numerator = 1.0 + 0.5 * halley * newton

            if (not np.isfinite(denominator) or abs(denominator) < 1e-14):
                
                raise FloatingPointError

            step = newton * numerator / denominator

            if not np.isfinite(step):
                raise FloatingPointError

            # Do not allow a step to reduce s by more than 50%.
            step = max(step, -0.5 * s)

            candidate = s + step

            # Safeguard: use bisection if the step leaves the bracket.
            if not (s_left < candidate < s_right):
                candidate = 0.5 * (s_left + s_right)

        except (FloatingPointError, OverflowError, ZeroDivisionError):
            candidate = 0.5 * (s_left + s_right)

        if abs(candidate - s) <= eps * max(s, 1.0):
            return candidate

        s = candidate

    # Final check before reporting non-convergence.
    residual = abs(normalized_black_call(x, s) - beta)

    if residual <= 10.0 * tol * max(beta, 1e-14):
        return s

    raise RuntimeError( f"Householder did not converge after {max_iter} iterations. " f"Residual = {residual:.3e}") 
    
    
def black_implied_volatility(price: float, F: float, K: float, T: float, discount_factor: float = 1.0, flag: str = "c", max_iter: int = 20, tol: float = 1e-13) -> float:
    """
    Black implied volatility for European calls and puts.

    Parameters
    ----------
    price : float
        Market option price, including discounting.
    F : float
        Forward price.
    K : float
        Strike.
    T : float
        Time to maturity in years.
    discount_factor : float
        Discount factor D(0,T).
    flag : str
        'c' for call, 'p' for put.

    Returns
    -------
    float
        Annualized implied volatility.
    """

    if flag not in ("c", "p"):
        raise ValueError("flag must be 'c' or 'p'.")

    if not all(np.isfinite(v) for v in (
        price, F, K, T, discount_factor
    )):
        raise ValueError("Inputs must be finite.")

    if F <= 0.0 or K <= 0.0 or T <= 0.0:
        raise ValueError("F, K and T must be strictly positive.")

    if discount_factor <= 0.0:
        raise ValueError("Discount factor must be positive.")

    if price < 0.0:
        raise ValueError("Option price cannot be negative.")

    # Convert to undiscounted price.
    undiscounted_price = price / discount_factor

    if flag == "c":
        intrinsic = max(F - K, 0.0)
        maximum_price = F
    else:
        intrinsic = max(K - F, 0.0)
        maximum_price = K

    if undiscounted_price < intrinsic:
        raise ValueError("Option price is below intrinsic value.")

    if undiscounted_price >= maximum_price:
        raise ValueError( "Option price is at or above its maximum; " "there is no finite implied volatility.")

    # Remove intrinsic value.
    time_value = undiscounted_price - intrinsic

    if time_value == 0.0:
        return 0.0

    # # Normalize the time value.
    # beta = time_value / np.sqrt(F * K)

    # # Map calls and puts to the OTM-call representation.
    # x = -abs(np.log(F / K))
    
    beta, x = normalize_black_option(price=price, F=F, K=K, discount_factor=discount_factor, flag=flag)

    # Solve for total volatility s = sigma * sqrt(T).
    s = solve_normalized_implied_volatility(beta=beta, x=x, max_iter=max_iter, tol=tol)
    
    return s / np.sqrt(T)