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
x = sympy.Symbol('x')



y1 = sympy.integrate(sympy.sin(sympy.pi*x))

y2 = sympy.integrate(sympy.exp(x))

y3 = sympy.integrate(sympy.erfc(x))

print(y1)
print(y2)
print(y3)

t1 = sympy.integrate(taylorExpansion(sympy.sin(sympy.pi*x),0,10))

t2 = sympy.integrate(taylorExpansion(sympy.exp(x),0,10))

t3 = sympy.integrate(taylorExpansion(sympy.erfc(x), 0, 10))

print(t1)
print(t2)
print(t3)

#32
sin = sympy.sin(sympy.pi*x)
e = sympy.exp(x)
erfc = sympy.erfc(x)

isin = []
order = []
for i in range (0,10):
    err_fun = sympy.lambdify( x, abs( sin - taylorExpansion(sin, 0, i) ) )
    isin.append( scipy.integrate.quad( err_fun, -1, 1 )[0] )
    order.append(i)


plt.plot( order, isin )
plt.yscale('log')
plt.show()

# sympy.plot(isin, show = True, yscale = 'logrithmic', xlim = [-1,1])

# isin = scipy.integrate.quad( err_fun, -1, 1 )[0]

# print( isin )
# ie = scipy.integrate(abs(e-taylorExpansion(e, 0, 10)))
# ierfc = scipy.integrate(abs(erfc-taylorExpansion(erfc, 0, 10)))

# sympy.plot(isin, show = True, yscale = 'logrithmic', xlim = [-1,1])
# sympy.plot(ie, show = True, yscale = 'logrithmic', xlim = [-1,1])
# sympy.plot(ierfc, show = True, yscale = 'logrithmic', xlim = [-2,2])



#34

## EQUIVALENT
sin_2pi = lambda x : math.sin( 2*math.pi * x )

def sin_2pi( x ):
   return math.sin( 2 * math.pi * x) 
## END EQUIVALENT
    
def evaluateMonomialBasis1D(p):
    x = sympy.Symbol('x')
    y = x**p
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
        
#33
#sympy.plot(evaluateMonomialBasis1D(0), xlim=[-1,1])
# evaluateMonomialBasis1D(1)
# evaluateMonomialBasis1D(2)
# evaluateMonomialBasis1D(3)
# evaluateMonomialBasis1D(4)
# evaluateMonomialBasis1D(5)
# evaluateMonomialBasis1D(6)
# evaluateMonomialBasis1D(7)
# evaluateMonomialBasis1D(8)
# evaluateMonomialBasis1D(9)
# evaluateMonomialBasis1D(10)