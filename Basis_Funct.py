# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:13:11 2022

@author: brian
"""

import sympy
import matplotlib.pyplot as plt
import numpy
import scipy
# import scipy.integrate
from scipy import integrate
import math

print( __name__ )

# 29
def taylorExpansion( fun, a, order ):
    x=sympy.Symbol('x')
    i = 0
    p = 0
    t = 0
    
    while i <= order + 1:
        p =(fun.diff(x, i).subs(x,a)/(sympy.factorial(i)))*(x-a)**i
        i += 1
        t += p
    return t

#31



# y1 = sympy.integrate(sympy.sin(sympy.pi*x))

# y2 = sympy.integrate(sympy.exp(x))

# y3 = sympy.integrate(sympy.erfc(x))

# # print(y1)
# # print(y2)
# # print(y3)

# t1 = sympy.integrate(taylorExpansion(sympy.sin(sympy.pi*x),0,10))

# t2 = sympy.integrate(taylorExpansion(sympy.exp(x),0,10))

# t3 = sympy.integrate(taylorExpansion(sympy.erfc(x), 0, 10))

# # print(t1)
# # print(t2)
# # print(t3)

# #32
# sin = sympy.sin(sympy.pi*x)
# e = sympy.exp(x)
# erfc = sympy.erfc(x)

# isin = []
# order = []
# for i in range (0,10):
#     err_fun = sympy.lambdify( x, abs( sin - taylorExpansion(sin, 0, i) ) )
#     isin.append( scipy.integrate.quad( err_fun, -1, 1 )[0] )
#     order.append(i)

# ie = []
# order = []
# for i in range (0,10):
#      err_fun = sympy.lambdify( x, abs( e - taylorExpansion(e, 0, i) ) )
#      ie.append( scipy.integrate.quad( err_fun, -1, 1 )[0] )
#      order.append(i)

# ierfc = []
# order = []
# for i in range (0,10):
#      err_fun = sympy.lambdify( x, abs( erfc - taylorExpansion(erfc, 0, i) ) )
#      ierfc.append( scipy.integrate.quad( err_fun, -1, 1 )[0] )
#      order.append(i)


# plt.plot( order, ierfc)
# plt.yscale('log')
# plt.show()



#34

## EQUIVALENT
sin_2pi = lambda x : math.sin( 2*math.pi * x )

def sin_2pi( x ):
   return math.sin( 2 * math.pi * x) 
## END EQUIVALENT
    
def evaluateMonomialBasis1D(degree, variate):
    y = variate**degree
    return y

import unittest

class Test_evaluateMonomialBasis1D( unittest.TestCase ):
   def test_basisAtBounds( self ):
       self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = 0, variate = 0 ), second = 1.0, delta = 1e-12 )
       for p in range( 1, 11 ):
           self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 0 ), second = 0.0, delta = 1e-12 )
           self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 1 ), second = 1.0, delta = 1e-12 )

   def test_basisAtMidpoint( self ):
       for p in range( 0, 11 ):
           self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 0.5 ), second = 1 / ( 2**p ), delta = 1e-12 )
        
sympy.Symbol('x')
sympy.plot(evaluateMonomialBasis1D(0,1), xlim = [-1,1], ylim = [-2,2])
plt.plot(evaluateMonomialBasis1D(1, 0))
# evaluateMonomialBasis1D(2)
# evaluateMonomialBasis1D(3)
# evaluateMonomialBasis1D(4)
# evaluateMonomialBasis1D(5)
# evaluateMonomialBasis1D(6)
# evaluateMonomialBasis1D(7)
# evaluateMonomialBasis1D(8)
# evaluateMonomialBasis1D(9)
# evaluateMonomialBasis1D(10)


#36
def evalLegendreBasis1D(degree, variate):
    if degree == 0:
        P = 1
    if degree == 1:
        P = variate
    else:
        x = sympy.Symbol('x')
        P1 = 1/(2**degree * math.factorial(degree))
        P2 = sympy.lambdify(x, sympy.diff((x, ((x**2 - 1)**degree)), degree))
        P = P1 * P2(variate)
    return P

class Test_evalLegendreBasis1D( unittest.TestCase ):
    def test_basisAtBounds( self ):
        for p in range( 0, 2 ):
            if ( p % 2 == 0 ):
                self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = -1 ), second = +1.0, delta = 1e-12 )
            else:
                self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = -1 ), second = -1.0, delta = 1e-12 )
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = +1 ), second = 1.0, delta = 1e-12 )

    def test_constant( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 0, variate = x ), second = 1.0, delta = 1e-12 )

    def test_linear( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 1, variate = x ), second = x, delta = 1e-12 )

    def test_quadratic_at_roots( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 2, variate = -1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 2, variate = +1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )

    def test_cubic_at_roots( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = -math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = +math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )
        
        
# 38
def evaluateLagrangeBasis1D(variate, degree, basis_idx):
    
    return

class Test_evaluateLagrangeBasis1D( unittest.TestCase ):
    def test_linearLagrange( self ):
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 1, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 1, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 1, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 1, basis_idx = 1 ), second = 1.0, delta = 1e-12 )

    def test_quadraticLagrange( self ):
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 2 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 1 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 2 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 2 ), second = 1.0, delta = 1e-12 )

#42
def evaluateBernsteinBasis1D(variate, degree, basis_idx):
    
    return

class Test_evaluateBernsteinBasis1D( unittest.TestCase ):
    def test_linearBernstein( self ):
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 1, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 1, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 1, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 1, basis_idx = 1 ), second = 1.0, delta = 1e-12 )

    def test_quadraticBernstein( self ):
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 0 ), second = 1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 1 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 2 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 0 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 1 ), second = 0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 2 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 1 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 2 ), second = 1.00, delta = 1e-12 )