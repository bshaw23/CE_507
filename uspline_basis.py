# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:16:12 2022

@author: brian
"""

import unittest
import math
import sys

import sympy
import numpy

import Basis as basis
import Gram_Matrix_Basis as gram
import bext
import sympy
import numpy
import uspline
import quadrature

# =============================================================================
# Evaluate the derivth derivative of the Berstein Basis
# =============================================================================

def evalBernsteinBasisDeriv(degree, basis_idx, deriv, domain, variate):
    x = sympy.Symbol('x')
    
    symboleval_bernbasis = basis.evalBernsteinBasis1D(x, degree, domain, basis_idx)
    
    if deriv == 0:
        eval_bernbasis = symboleval_bernbasis.subs(x, variate)
    else:
        deriv_bernbasis = sympy.diff(symboleval_bernbasis, x, deriv)
        eval_bernbasis = deriv_bernbasis.subs(x, variate)
    return eval_bernbasis

# =============================================================================
# Evaluate the derivth derivative of the spline basis
# =============================================================================
def evalSplineBasisDeriv1D(extraction_operator, basis_idx, deriv, domain, variate):
    degree = int(extraction_operator.shape[0]-1)
    bernbasis = numpy.zeros((degree+1))
    for i in range(0, degree +1):
        bernbasis[i] = evalBernsteinBasisDeriv(degree, i, deriv, domain, variate)
    eval_splinebasis = numpy.dot(extraction_operator, bernbasis)
    return eval_splinebasis[basis_idx]

# =============================================================================
# Assemble Stiffness Matrix
# =============================================================================
def assembleStiffnessMatrix(problem, uspline_bext):
    E = problem["elastic_modulus"]
    A = problem["area"]
    num_elems = bext.getNumElems(uspline_bext)
    K_list = []
    
    for i in range(0,num_elems):
        degree = bext.getElementDegree(uspline_bext, i)
        n = int(numpy.ceil((2*degree+1)/2))
        qp,w = quadrature.computeGaussLegendreQuadrature(n)
        domain = bext.getElementDomain(uspline_bext, i)
        extraction_operator = bext.getElementExtractionOperator(uspline_bext, i)
        
        K = numpy.zeros((degree+1,degree+1))
        jacob = (gram.jacobian([-1,1], domain))**(-1)
        for a in range(degree+1):
            for b in range(degree+1):
                for q in range(len(qp)):
                    N_A = evalSplineBasisDeriv1D(extraction_operator, a, 1, [-1,1], qp[q])
                    N_B = evalSplineBasisDeriv1D(extraction_operator, b, 1, [-1,1], qp[q])
                    K[a,b] += E * A * N_A * N_B * w[q] * jacob
                    
        K_list.append(K)
        
    K = local_to_globalK(K_list, uspline_bext)
    return K

def local_to_globalK(K_list, uspline_bext):
    dim = bext.getNumNodes(uspline_bext)
    K = numpy.zeros((dim,dim))
    for k in range(len(K_list)):
        degree = bext.getElementDegree(uspline_bext, k)
        for a in range(0, degree + 1):
            A = bext.getElementNodeIds(uspline_bext, k)[a]
            for b in range(0, degree + 1):
                B = bext.getElementNodeIds(uspline_bext, k)[b]
                K[A,B] += K_list[k][a,b]
                
    return K

# =============================================================================
# Assemble Force Vector 
# =============================================================================
def assembleForceVector(target_fun, node_coords, ien_array, solution_basis):
    num_elems = len(ien_array)
    degree_list = []
    F_list = []
    
    for i in range(0,num_elems):
        degree = int(len(ien_array[i])-1)
        degree_list.append(degree)
        n = int(numpy.ceil((2*degree+1)/2))
        xi_pq, w_pq = quadrature.computeGaussLegendreQuadrature(n)
        domain = [node_coords[i][0],node_coords[i][-1]]
        
        F_term = numpy.zeros((degree+1))
        for A in range(0,degree+1):
            for q in range(0,len(xi_pq)):
                if solution_basis == basis.evalBernsteinBasis1D:
                    N_A = basis.evalBernsteinBasis1D(xi_pq[q], degree, [-1,1], A)
                    jacob = gram.jacobian([-1,1], domain)
                elif solution_basis == basis.evalLegendreBasis1D:
                    N_A = basis.evalLegendreBasis1D(A, xi_pq[q])
                    jacob = gram.jacobian([-1,1], domain)
                elif solution_basis == basis.evaluateLagrangeBasis1D:
                    N_A = basis.evaluateLagrangeBasis1D(xi_pq[q], degree, A)
                    jacob = gram.jacobian([-1,1], domain)
                
                F_term[A] += N_A * target_fun(gram.change_of_coords([-1,1],domain,xi_pq[q])) * w_pq[q] * jacob 
        F_list.append(F_term)
    F = Local_to_Global_F_Vector(F_list,degree_list)
    return F

# =============================================================================
#  Local to Global F Vector
# =============================================================================
def Local_to_Global_F_Vector(F_list,degree):
    dim = sum(degree) + 1
    F = numpy.zeros((dim))
    
    for i in range(0,len(F_list)):
        for a in range(0,degree[i]+1):
            A = i * (degree[i]) + a
            F[A] += F_list[i][a]
    
    return F

# class Test_evalBernsteinBasisDeriv( unittest.TestCase ):
#         def test_constant_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.0, delta = 1e-12 )

#         def test_constant_1st_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )

#         def test_constant_2nd_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )

#         def test_linear_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.0, delta = 1e-12 )

#         def test_linear_at_gauss_pts( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate =  0.5 ), second = 0.5, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate =  0.5 ), second = 0.5, delta = 1e-12 )

#         def test_quadratic_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.25, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.50, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.00, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.25, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.00, delta = 1e-12 )

#         def test_quadratic_at_gauss_pts( self ):
#               x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
#               x = [ gram.change_of_coords( [-1, 1], [0, 1], xi ) for xi in x ]
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.62200846792814620, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.04465819873852045, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.33333333333333333, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.33333333333333333, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.04465819873852045, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.62200846792814620, delta = 1e-12 )

#         def test_linear_1st_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +1.0, delta = 1e-12 )

#         def test_linear_1st_deriv_at_gauss_pts( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second = +1.0, delta = 1e-12 )

#         def test_linear_2nd_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0, delta = 1e-12 )

#         def test_linear_2nd_deriv_at_gauss_pts( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.5 ), second = 0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 0.5 ), second = 0, delta = 1e-12 )

#         def test_quadratic_1st_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.0 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +2.0, delta = 1e-12 )

#         def test_quadratic_1st_deriv_at_gauss_pts( self ):
#               x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
#               x = [ gram.change_of_coords( [-1, 1], [0, 1], xi ) for xi in x ]
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = x[0] ), second = -1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = x[1] ), second = -1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = x[0] ), second = +2.0 / math.sqrt(3), delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = x[1] ), second = -2.0 / math.sqrt(3), delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = x[0] ), second = +1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = x[1] ), second = +1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )

#         def test_quadratic_2nd_deriv_at_nodes( self ):
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.0 ), second =  0.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  1.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +2.0, delta = 1e-12 )

#         def test_quadratic_2nd_deriv_at_gauss_pts( self ):
#               x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
#               x = [ gram.change_of_coords( [-1, 1], [0, 1], xi ) for xi in x ]
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, domain = [0, 1], variate = x[0] ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, domain = [0, 1], variate = x[1] ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, domain = [0, 1], variate = x[0] ), second = -4.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, domain = [0, 1], variate = x[1] ), second = -4.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, domain = [0, 1], variate = x[0] ), second = +2.0, delta = 1e-12 )
#               self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, domain = [0, 1], variate = x[1] ), second = +2.0, delta = 1e-12 )
              
# class Test_evalSplineBasisDeriv1D( unittest.TestCase ):
#         def test_C0_linear_0th_deriv_at_nodes( self ):
#               C = numpy.eye( 2 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 1.0 )

#         def test_C0_linear_1st_deriv_at_nodes( self ):
#               C = numpy.eye( 2 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = -1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = -1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

#         def test_C1_quadratic_0th_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.25 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.625 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.125 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.5 )

#         def test_C1_quadratic_1st_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = -2.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +2.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = -1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = +0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = +0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = -1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

#         def test_C1_quadratic_2nd_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = +2.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = -3.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = +1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = +2.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = -3.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = +1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = +2.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = -3.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

#         def test_biunit_C0_linear_0th_deriv_at_nodes( self ):
#               C = numpy.eye( 2 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 1.0 )

#         def test_biunit_C0_linear_1st_deriv_at_nodes( self ):
#               C = numpy.eye( 2 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = -0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = -0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.5 )

#         def test_biunit_C1_quadratic_0th_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.25 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.625 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.125 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.5 )

#         def test_biunit_C1_quadratic_1st_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = -1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +1.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = -0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.0 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = -0.5 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.5 )

#         def test_biunit_C1_quadratic_2nd_deriv_at_nodes( self ):
#               C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = +0.50 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = -0.75 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = +0.25 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = +0.50 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = -0.75 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = +0.50 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = -0.75 )
#               self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = +0.25 )
              
class test_assembleStiffnessMatrix( unittest.TestCase ):
       def test_one_linear_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1 ], "continuity": [ -1, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [ 1.0, -1.0 ], [ -1.0, 1.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_linear_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [ 2.0, -2.0, 0.0 ], [ -2.0, 4.0, -2.0 ], [ 0.0, -2.0, 2.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_one_quadratic_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2 ], "continuity": [ -1, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  4.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0,  4.0 / 3.0, -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0, -2.0 / 3.0,  4.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_quadratic_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 0, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  8.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0,  0.0,        0.0 ],
                                                 [ -4.0 / 3.0,  8.0 / 3.0, -4.0 / 3.0,  0.0,        0.0 ],
                                                 [ -4.0 / 3.0, -4.0 / 3.0, 16.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0 ],
                                                 [  0.0,        0.0,       -4.0 / 3.0,  8.0 / 3.0, -4.0 / 3.0 ],
                                                 [  0.0,        0.0,       -4.0 / 3.0, -4.0 / 3.0,  8.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_quadratic_C1_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  8.0 / 3.0, -2.0,       -2.0/ 3.0,   0.0 ],
                                                 [ -2.0,        8.0 / 3.0,  0.0,       -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0,  0.0,        8.0 / 3.0, -2.0 ],
                                                 [  0.0,       -2.0 / 3.0, -2.0,        8.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

unittest.main()