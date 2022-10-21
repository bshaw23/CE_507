# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:17:26 2022

@author: brian
"""
import unittest
import numpy
import sympy
import math
import Basis as basis
import scipy
import mesh

def generateMesh1D(xmin, xmax, num_elems, degree):
    # create the node coordinates vecctor
    node_coords = numpy.linspace(xmin, xmax, (num_elems*degree) + 1)
    #create the element ID and nodal array
    ien_array = numpy.zeros((num_elems, degree + 1))
    for i in range(0, num_elems):
        local_elem = []
        for j in range(0, degree + 1):
            local_elem.append(i * degree + j)
        ien_array[i,:] = local_elem
    return node_coords, ien_array

def computeSolution(target_fun, domain, num_elems, degree):
    node_coords, ien_array = generateMesh1D(domain[0], domain[1], num_elems, degree)
    sol_coeffs = []
    for i in range(0, len(node_coords)):
        x = target_fun(node_coords[i])
        sol_coeffs.append(x)
    return sol_coeffs

def evaluateSolutionAt (x, coeff, node_coords, ien_array, eval_basis):
    degree = (len(node_coords)-1)/len(ien_array)
    elem_idx = mesh.getElementIdxContainingPoint(x, node_coords, ien_array)
    elem_nodes = mesh.getElementNodes(ien_array, elem_idx)
    elem_domain = mesh.getElementDomain(ien_array, node_coords, elem_idx)
    param_coord = mesh.spatialToParamCoords(x, elem_domain)
    sol_at_point = 0
    for i in range(0,len(elem_nodes)):
        curr_node = elem_nodes[i]
        sol_at_point += coeff[curr_node]*eval_basis(param_coord, degree, i) #param_coord, (len(node_coords)-1)/len(ien_array), i
    return sol_at_point

def computeFitError( target_fun, coeff, node_coords, ien_array, eval_basis ):
    num_elems = ien_array.shape[0]
    domain = [ min( node_coords ), max( node_coords ) ]
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ) )
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    return fit_error, residual

# class Test_generateMesh1D( unittest.TestCase ):
#     def test_make_1_linear_elem( self ):
#         gold_node_coords = numpy.array( [ 0.0, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 1, degree = 1 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
#     def test_make_1_quadratic_elem( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1, 2 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 1, degree = 2 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
#     def test_make_2_linear_elems( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1 ], [ 1, 2 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 2, degree = 1 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
#     def test_make_2_quadratic_elems( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1, 2 ], [ 2, 3, 4 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 2, degree = 2 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
#     def test_make_4_linear_elems( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 4, degree = 1 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
    
#     def test_make_4_quadratic_elems( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1, 2 ], [ 2, 3, 4 ], [ 4, 5, 6 ], [ 6, 7, 8 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 4, degree = 2 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
        
# class Test_computeSolution( unittest.TestCase ):
#     def test_single_linear_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x, domain = [-1.0, 1.0 ], num_elems = 1, degree = 1 )
#         gold_solution = numpy.array( [ -1.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
    
#     def test_single_quad_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 1, degree = 2 )
#         gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
    
#     def test_two_linear_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 2, degree = 1 )
#         gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
    
#     def test_four_quad_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 4, degree = 1 )
#         gold_solution = numpy.array( [ 1.0, 0.25, 0.0, 0.25, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
        
class Test_evaluateSolutionAt( unittest.TestCase ):
    def test_single_linear_element( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, 1, 1 )
        coeff = numpy.array( [-1.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = -1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +1.0 )
    
    def test_two_linear_elements( self ):
        node_coords, ien_array = mesh.generateMesh1D( -1, 1, 2, 1 )
        coeff = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +1.0 )
    
    def test_single_quadratic_element( self ):
        node_coords, ien_array = mesh.generateMesh1D( -1, 1, 1, 2 )
        coeff = numpy.array( [+1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +1.0 )
    
    def test_two_quadratic_elements( self ):
        node_coords, ien_array = mesh.generateMesh1D( -2, 2, 2, 2 )
        coeff = numpy.array( [ 1.0, 0.25, 0.5, 0.25, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +1.00 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +0.25 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +0.50 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +0.25 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evaluateLagrangeBasis1D ), second = +1.00 )
        