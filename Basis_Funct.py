# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:13:11 2022

@author: brian
"""

import sympy
import matplotlib.pyplot as plt
import numpy

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

#30





#34

def evaluateMonomialBasis1D(p):
    x = sympy.Symbol('x')
    y = x**p
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()
    return

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
# evaluateMonomialBasis1D(0)
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