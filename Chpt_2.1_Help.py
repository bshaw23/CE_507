# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:42:19 2022

@author: brian
"""
import Basis_Funct as BF

import numpy as np
import sympy
import matplotlib.pyplot as plt

x = sympy.Symbol('x')

fun = sympy.sin(sympy.pi*x)
#Tx = BF.taylorExpansion(fun, 0, 10)
#print(Tx)

sympy.plot(sympy.sin(sympy.pi*x), BF.taylorExpansion(fun, 0, 0), BF.taylorExpansion(fun, 0, 1), \
           BF.taylorExpansion(fun, 0, 3), BF.taylorExpansion(fun, 0, 5), \
           BF.taylorExpansion(fun, 0, 7), show = True, xlim=[-1,1], ylim=[-2,2])
    
    