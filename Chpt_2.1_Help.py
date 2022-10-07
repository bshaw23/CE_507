# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:42:19 2022

@author: brian
"""
import Basis_Funct as BF
import Basis as b

import numpy as np
import sympy
import matplotlib.pyplot as plt

x = sympy.Symbol('x')

#30
fun = sympy.erfc(x)
#Tx = BF.taylorExpansion(fun, 0, 10)
#print(Tx)



# sympy.plot(sympy.sin(sympy.erfc(x)), BF.taylorExpansion(fun, 0, 0), BF.taylorExpansion(fun, 0, 1), \
           # BF.taylorExpansion(fun, 0, 2), BF.taylorExpansion(fun, 0, 3), \
           # BF.taylorExpansion(fun, 0, 4), show = True, xlim=[-1,1], ylim=[0,3])
    
# sympy.plot(x**3 + x**2 + x +1, x**3, x**2, x, 1, show = True, xlim = [-1,1], ylim = [-2.5, 7.5])   

# sympy.plot(2*b.evaluateLagrangeBasis1D(x, 1, 0) + 6*b.evaluateLagrangeBasis1D(x, 1,1), 2*b.evaluateLagrangeBasis1D(x, 1, 0),\
           # 6*b.evaluateLagrangeBasis1D(x, 1,1), xlim = [-1,1], ylim = [0,6])

# sympy.plot(4*b.evalLegendreBasis1D(0, x)+2*b.evalLegendreBasis1D(1, x), 4*b.evalLegendreBasis1D(0, x), \
           # 2 * b.evalLegendreBasis1D(1, x) ,xlim = [-1,1], ylim = [0,6])
    
# sympy.plot(2*b.evaluateBernsteinBasis1D(x, 1, 0) + 6*b.evaluateBernsteinBasis1D(x, 1, 1), 2*b.evaluateBernsteinBasis1D(x, 1, 0),\
#            6*b.evaluateBernsteinBasis1D(x, 1,1), show = True, xlim = [-1,1], ylim = [0,6])

# sympy.plot(3/2*b.evaluateLagrangeBasis1D(x, 2, 0)+5/2*b.evaluateLagrangeBasis1D(x, 2, 1) + 15/2*b.evaluateLagrangeBasis1D(x, 2, 2), \
#            3/2*b.evaluateLagrangeBasis1D(x, 2, 0), 5/2*b.evaluateLagrangeBasis1D(x, 2, 1),\
#            15/2*b.evaluateLagrangeBasis1D(x, 2, 2), xlim = [-1,1], ylim = [0,7])

# sympy.plot(19/6*b.evalLegendreBasis1D(0, x)+3*b.evalLegendreBasis1D(1, x)+4/3*b.evalLegendreBasis1D(2, x), \
#            19/6*b.evalLegendreBasis1D(0, x), 3*b.evalLegendreBasis1D(1, x), 4/3*b.evalLegendreBasis1D(2, x), xlim=[-1,1], ylim = [-2.5,7.5])
    
# sympy.plot(3/2*b.evaluateBernsteinBasis1D(x, 2, 0) + 1/2*b.evaluateBernsteinBasis1D(x, 2, 1)+ \
#            4/3*b.evaluateBernsteinBasis1D(x, 2, 2), 3/2*b.evaluateBernsteinBasis1D(x, 2, 0), \
#                1/2*b.evaluateBernsteinBasis1D(x, 2, 1), 4/3*b.evaluateBernsteinBasis1D(x, 2, 2), xlim = [-1,1], ylim = [0,7])

sympy.plot(20/27*b.evaluateLagrangeBasis1D(x, 3, 1) + 40/27*b.evaluateLagrangeBasis1D(x, 3, 2) +\
           4*b.evaluateLagrangeBasis1D(x, 3, 3), 20/27*b.evaluateLagrangeBasis1D(x, 3, 1), \
               40/27*b.evaluateLagrangeBasis1D(x, 3, 2), 4*b.evaluateLagrangeBasis1D(x, 3, 3), \
                   xlim = [-1,1], ylim = [-1,4])
    
sympy.plot(4/3*b.evalLegendreBasis1D(0, x)+8/5*b.evalLegendreBasis1D(1, x)+2/3*b.evalLegendreBasis1D(2, x),\
           +2/3*b.evalLegendreBasis1D(3, x), 4/3*b.evalLegendreBasis1D(0, x), 8/5*b.evalLegendreBasis1D(1, x),\
           2/3*b.evalLegendreBasis1D(2, x), 2/3*b.evalLegendreBasis1D(3, x), xlim = [-1,1], ylim = [-1,4])
    
sympy.plot(4/3*b.evaluateBernsteinBasis1D(x, 3, 1)+4*b.evaluateBernsteinBasis1D(x, 3, 3),\
           4/3*b.evaluateBernsteinBasis1D(x, 3, 1), 4*b.evaluateBernsteinBasis1D(x, 3, 3), \
               xlim = [-1,1], ylim = [0,4])