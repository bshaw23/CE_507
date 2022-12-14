# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:27:35 2022

@author: brian
"""
import numpy
import unittest
import matplotlib.pyplot as plt
import scipy
import Basis as basis
import mesh 
import Gram_Matrix_Basis as gram
import Galerkin_OneElement as one_elem

def computeSolution(target_fun, domain, degree, solution_basis):
    xmin = domain[0]
    xmax = domain[1]
    node_coords, ien_array = mesh.generateMesh(xmin, xmax, degree)
    M = assembleGramMatrix(node_coords, ien_array, solution_basis)
    F = assembleForceVector(target_fun, ien_array, node_coords, solution_basis)
    F = F.transpose()
    d = numpy.linalg.solve(M,F)
    #solution = d dotted with basis
    print (d)
    return d, node_coords, ien_array


def Local_to_Global_GramMatrix(M_list, degree):
    num_elems = len(M_list)
    dim = sum(degree)+1
    M = numpy.zeros((dim,dim))
    
    for i in range(num_elems):
        for a in range(degree[i]+1):
            A = i * (degree[i]) + a
            for b in range(degree[i]+1):
                B = i * (degree[i]) + b
                M[A,B] += M_list[i][a,b]
    return M

def Local_to_Global_FVector(F_list, degree):
    num_elems = len(F_list)
    dim = sum(degree)+1
    F = numpy.zeros(dim)
    
    for i in range(num_elems):
        for a in range(degree[i]+1):
            A = i * (degree[i]) + a
            F[A] += F_list[i][a]
    return F
    
def assembleGramMatrix(node_coords, ien_array, solution_basis):
    num_elems = len(ien_array)
    M_list = []
    degree_list = []
    for i in range(0, num_elems):
        degree = int(len(ien_array[i]) - 1)
        degree_list.append(degree)
        domain = [node_coords[i][0], node_coords[i][-1]]
        M = one_elem.assembleGramMatrix(domain, degree, solution_basis)
        
        M_list.append(M)
            
    M = Local_to_Global_GramMatrix(M_list, degree_list)
    return M

def assembleForceVector(target_fun, ien_array, node_coords, solution_basis):
    num_elems = len(ien_array)
    F_list = []
    degree_list = []
    for i in range(num_elems):
        degree = int(len(ien_array[i]) - 1)
        degree_list.append(degree)
        domain = [node_coords[i][0], node_coords[i][-1]]
        F = one_elem.assembleForceVector(target_fun, domain, degree, solution_basis)
        
        F_list.append(F)
        
    F = Local_to_Global_FVector(F_list, degree_list)
    return F

def evaluateSolutionAt( x, d_vector, node_coords, ien_array, solution_basis ):
    y = 0.0
    domain = [node_coords[0][0], node_coords[-1][-1]]
    num_elems = len(ien_array)
    
    elem_bound = numpy.linspace(domain[0], domain[1], num_elems + 1)
    for i in range(num_elems):
        if x>= elem_bound[i] and x <= elem_bound[i+1]:
            elem_idx = i
        else:
            continue
    
    degree = len(ien_array[elem_idx]) - 1
    elem_domain = [node_coords[elem_idx][0], node_coords[elem_idx][-1]]
    new_x = gram.change_of_coords(elem_domain, [-1,1], x)
    for i in range(0, len(ien_array[elem_idx])):
        d_idx = int(ien_array[elem_idx][i])
        if solution_basis == basis.evalLegendreBasis1D:
            basis_vec = solution_basis(i, new_x)
        else: 
            basis_vec = solution_basis(new_x, degree, i)
            
        y += basis_vec * d_vector[d_idx]
    return y

def computeFitError(target_fun, d_vector, node_coords, ien_array, solution_basis):
    num_elems = len(ien_array)
    xmin = node_coords[0][0]
    xmax = node_coords[-1][-1]
    domain = [xmin, xmax]
    abs_err_fun = lambda x: abs(target_fun(x) - evaluateSolutionAt( x, d_vector, node_coords, ien_array, solution_basis))
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    
    return fit_error, residual


def plotCompareGoldTestSolution2( gold_coeff, d_vector, node_coords, ien_array, solution_basis ):
    domain = [node_coords[0][0], node_coords[-1][-1]]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], d_vector, node_coords, ien_array, solution_basis )
        yt[i] = evaluateSolutionAt( x[i], d_vector, node_coords, ien_array, solution_basis )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()

def plotCompareFunToTestSolution2( fun, d_vector, node_coords, ien_array, solution_basis ):
    domain = [node_coords[0][0], node_coords[-1][-1]]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = fun( x[i] )
        yt[i] = evaluateSolutionAt( x[i], d_vector, node_coords, ien_array, solution_basis )
    plt.plot( x, y )
    plt.plot( x, yt )
    plt.show()

class Test_computeSolution( unittest.TestCase ):
    def test_cubic_polynomial_target( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        domain = [ 0, 1 ]
        degree = [2]*2
        solution_basis = basis.evalBernsteinBasis1Dold
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis =solution_basis )
        gold_sol_coeff = numpy.array( [ 1.0 / 120.0, 9.0 / 80.0, 1.0 / 40.0, -1.0 / 16.0, -1.0 / 120.0 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareGoldTestSolution2( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareFunToTestSolution2( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )

    def test_sin_target( self ):
        # print( "SIN TEST" )
        target_fun = lambda x: numpy.sin( numpy.pi * x )
        domain = [ 0, 1 ]
        degree = [2]*2
        solution_basis = basis.evalBernsteinBasis1Dold
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ -0.02607008, 0.9185523, 1.01739261, 0.9185523, -0.02607008 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareGoldTestSolution2( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareFunToTestSolution2( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )
        
    def test_erfc_target( self ):
        # print( "ERFC TEST" )
        target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
        domain = [ -2, 2 ]
        degree = [3]*2
        solution_basis = basis.evalBernsteinBasis1Dold
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ 1.98344387, 2.0330054, 1.86372084, 1., 0.13627916, -0.0330054, 0.01655613 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareGoldTestSolution2( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareFunToTestSolution2( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )
    
    def test_exptx_target( self ):
        # print( "EXPT TEST" )
        target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
        domain = [ -1, 1 ]
        degree = [5]*2
        solution_basis = basis.evalBernsteinBasis1Dold
        test_sol_coeff, node_coords, ien_array = computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = ( [ -1.00022471, -1.19005562, -0.9792369, 0.70884334, 1.73001439, 0.99212064, 0.44183573, 0.87014465, 0.5572111, 0.85241908, 0.99175228 ] )
        abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareGoldTestSolution2( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        plotCompareFunToTestSolution2( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )

class Test_assembleGramMatrix( unittest.TestCase ):
    def test_linear_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [1/15, 1/30, -1/60, 0, 0 ], [1/30, 4/15, 1/30, 0, 0], [-1/60, 1/30, 2/15, 1/30, -1/60], [ 0, 0, 1/30, 4/15, 1/30], [0, 0, -1/60, 1/30, 1/15] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [ 0.03809524,  0.02946429, -0.01071429,  0.00565476,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [ 0.02946429,  0.19285714, -0.02410714, -0.01071429,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [-0.01071429, -0.02410714,  0.19285714,  0.02946429,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [ 0.00565476, -0.01071429,  0.02946429,  0.07619048,  0.02946429, -0.01071429,  0.00565476 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000,  0.02946429,  0.19285714, -0.02410714, -0.01071429 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000, -0.01071429, -0.02410714,  0.19285714,  0.02946429 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000,  0.00565476, -0.01071429,  0.02946429,  0.03809524 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_linear_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1Dold )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1Dold )
        gold_gram_matrix = numpy.array( [ [1/10, 1/20, 1/60, 0, 0 ], [1/20, 1/15, 1/20, 0, 0 ], [1/60, 1/20, 1/5, 1/20, 1/60], [0, 0, 1/20, 1/15, 1/20], [0, 0, 1/60, 1/20, 1/10] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1Dold )
        gold_gram_matrix = numpy.array( [ [1/14, 1/28, 1/70, 1/280, 0, 0, 0 ], [1/28, 3/70, 9/280, 1/70, 0, 0, 0 ], [1/70, 9/280, 3/70, 1/28, 0, 0, 0 ], [1/280, 1/70, 1/28, 1/7, 1/28, 1/70, 1/280], [0, 0, 0, 1/28, 3/70, 9/280, 1/70], [0, 0, 0, 1/70, 9/280, 3/70, 1/28], [0, 0, 0, 1/280, 1/70, 1/28, 1/14 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
        
        
class Test_assembleForceVector( unittest.TestCase ):
    def test_lagrange_const_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, numpy.pi / 8.0, 3.0 * numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, numpy.pi / 16.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_linear_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: 2*x + numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ 0.20468287, 0.62654862, 0.73904862, 0.51769908, 0.81404862, 0.92654862, 0.31301621 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_quadratic_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: x**2.0
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ 1.04166667e-03, 0, 2.81250000e-02, 3.33333333e-02, 6.56250000e-02, 1.50000000e-01, 5.52083333e-02 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_lagrange_const_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1Dold )
        gold_force_vector = numpy.array( [ numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 4.0, numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 8.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_linear_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: 2*x + numpy.pi
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1Dold)
        gold_force_vector = numpy.array( [ 0.41769908, 0.44269908, 0.46769908, 1.03539816, 0.56769908, 0.59269908, 0.61769908 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_quadratic_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: x**2.0
        node_coords, ien_array = mesh.generateMesh( domain[0], domain[1], degree )
        test_force_vector = assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1Dold )
        gold_force_vector = numpy.array( [ 1/480, 1/160, 1/80, 1/15, 1/16, 13/160, 49/480 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
        
def plotCompareGoldTestSolution( gold_coeff, test_coeff, domain, solution_basis ):
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], domain, gold_coeff, solution_basis )
        yt[i] = evaluateSolutionAt( x[i], domain, test_coeff, solution_basis )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()

def plotCompareFunToTestSolution( fun, test_coeff, domain, solution_basis ):
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = fun( x[i] )
        yt[i] = evaluateSolutionAt( x[i], domain, test_coeff, solution_basis )
    plt.plot( x, y )
    plt.plot( x, yt )
    plt.show()
    
unittest.main()