# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:57:10 2026

@author: Nicola
"""

import numpy as np
from numerical_methods.numerical_methods import * 

import pytest

class TestRootFinding:
    
    def test_bisection_1(self):
        
        fun = lambda x : (x+5) / (x**2 -4*x +3) 
        a = -7 
        b = -3 
        
        desired = -5 
        actual = bisection_method(fun, a, b)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_bisection_2(self):
        
        fun = lambda x : (9*x**2 - 9*x) * (x**3 + 8) / (x**2 + 2) 
        # a = -3
        # b = -1
        
        desired1 = -2 
        desired2 = 0
        desired3 = 1
        
        actual1 = bisection_method(fun, -3, -1)
        actual2 = bisection_method(fun, -1, .5)
        actual3 = bisection_method(fun, .5, 2)
        
        np.testing.assert_allclose(actual1, desired1, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(actual2, desired2, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(actual3, desired3, atol=1e-3, rtol=1e-3) 
    
    def test_newton_1(self):
        
        fun = lambda x : (x+5) / (x**2 -4*x +3) 
        dfun = lambda x : (fun(x + 1e-5) - fun(x)) / 1e-5
        a = -7 
        
        desired = -5 
        actual = newton_method(fun, dfun, a)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_newton_2(self):
        
        fun = lambda x : (9*x**2 - 9*x) * (x**3 + 8) / (x**2 + 2) 
        dfun = lambda x : (fun(x + 1e-5) - fun(x)) / 1e-5
        # a = -3
        # b = -1
        
        desired1 = -2 
        desired2 = 0
        desired3 = 1
        
        actual1 = newton_method(fun, dfun, -3)
        actual2 = newton_method(fun, dfun, -.5)
        actual3 = newton_method(fun, dfun, .5)
        
        np.testing.assert_allclose(actual1, desired1, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(actual2, desired2, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(actual3, desired3, atol=1e-3, rtol=1e-3) 
    
    def test_secant_1(self):
        
        fun = lambda x : (x+5) / (x**2 -4*x +3) 
        a = -7 
        b = -3 
        
        desired = -5 
        actual = secant_method(fun, a, b)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_secant_2(self):
        
        fun = lambda x : (9*x**2 - 9*x) * (x**3 + 8) / (x**2 + 2) 
        # a = -3
        # b = -1
        
        desired1 = -2 
        desired2 = 0
        desired3 = 1
        
        actual1 = secant_method(fun, -3, -1)
        actual2 = secant_method(fun, -1, .5)
        actual3 = secant_method(fun, .5, 2)
        
        np.testing.assert_allclose(actual1, desired1, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(actual2, desired2, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(actual3, desired3, atol=1e-3, rtol=1e-3) 



class TestIntegration1var:
    
    def test_trapezoidal1(self):
        
        fun = lambda x : 1 / x
        a = 1 
        b = 2 
        
        desired = np.log(2)
        actual = trapezoidal_rule(fun, a, b, 250)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_trapezoidal2(self):
        
        fun = lambda x : 1 / x**2
        a = 1 
        b = 2 
        
        desired = .5
        actual = trapezoidal_rule(fun, a, b, 250)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_trapezoidal3(self):
        
        fun = lambda x : x**2 - 2/x**2
        a = 1 
        b = 2 
        
        desired = 4/3
        actual = trapezoidal_rule(fun, a, b, 250)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_trapezoidal4(self):
        
        fun = lambda x : np.exp(x)
        a = 1 
        b = 2 
        
        desired = np.exp(b) - np.exp(a)
        actual = trapezoidal_rule(fun, a, b, 250)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_midpoint1(self):
        
        fun = lambda x : 1 / x
        a = 1 
        b = 2 
        
        desired = np.log(2)
        actual = midpoint_rule(fun, a, b, 500)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_midpoint2(self):
        
        fun = lambda x : 1 / x**2
        a = 1 
        b = 2 
        
        desired = .5
        actual = midpoint_rule(fun, a, b, 500)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_midpoint3(self):
        
        fun = lambda x : x**2 - 2/x**2
        a = 1 
        b = 2 
        
        desired = 4/3
        actual = midpoint_rule(fun, a, b, 750)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_midpoint4(self):
        
        fun = lambda x : np.exp(x)
        a = 1 
        b = 2 
        
        desired = np.exp(b) - np.exp(a)
        actual = midpoint_rule(fun, a, b, 1000)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_rectangle1(self):
        
        fun = lambda x : 1 / x
        a = 1 
        b = 2 
        
        desired = np.log(2)
        actual = rectangle_rule(fun, a, b, 500)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_rectangle2(self):
        
        fun = lambda x : 1 / x**2
        a = 1 
        b = 2 
        
        desired = .5
        actual = rectangle_rule(fun, a, b, 500)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_rectangle3(self):
        
        fun = lambda x : x**2 - 2/x**2
        a = 1 
        b = 2 
        
        desired = 4/3
        actual = rectangle_rule(fun, a, b, 1000)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
    
    def test_rectangle4(self):
        
        fun = lambda x : np.exp(x)
        a = 1 
        b = 2 
        
        desired = np.exp(b) - np.exp(a)
        actual = rectangle_rule(fun, a, b, 500)
        
        np.testing.assert_allclose(actual, desired, atol=1e-3, rtol=1e-3)
        
        
    
        
    
        
        