# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:02:58 2022

@author: brian
"""
import unittest
import numpy

# one dimensional mesh generation for varying degree
def generateMesh(xmin, xmax, degree):
    num_elems = len(degree)
    elem_boundaries_coords = numpy.linspace( xmin, xmax, num_elems+1)
    ien_array = {}
    node_coords = []
    for elem_idx in range(0, num_elems):
        elem_degree = degree[elem_idx]
        num_elem_nodes = elem_degree + 1
        elem_verts = [elem_idx, elem_idx + 1]
        elem_node_coords = numpy.linspace(elem_boundaries_coords[elem_verts[0] ], elem_boundaries_coords[elem_verts[-1]])
        if elem_idx == 0:
            ien_array[elem_idx] = list(range(0,num_elem_nodes))
        else:
            start_node_idx = ien_array[elem_idx -1][-1]
            stop_node_idx = ien_array[elem_idx -1][-1] + num_elem_nodes
            ien_array[elem_idx] = list(range(start_node_idx, stop_node_idx))
            node_coords.append(elem_node_coords[1:])
    return node_coords, ien_array

# def computeElemBoundaryCoords( xmin, xmax, num_elems):
#     elem_boundaries_coords = numpy.linspace( xmin, xmax, num_elems+1)
#     return elem_boundaries_coords

# class Test_compute ElemBoundaryCoords( unittest.TestCase):
#     def test_single_element_unit_domain(self):
#         gold_elem_boundary_coords = numby.array([0,1])
#         test_elem_boundary_coords = computeElemBoundaryCoords(xmin = 0, xmax = 1, num_elems = 1)
#         self.assertTrue( numpy.allclose( gold_elem_boundary_coords, test_elem_boundary_coords))
        
#     def test_single_element_shifted_domain(self):
#         gold_elem_boundary_coords = numby.array([0.28374,2.87234])
#         test_elem_boundary_coords = computeElemBoundaryCoords(xmin = 0.28374, xmax = 2.87234, num_elems = 1)
#         self.assertTrue( numpy.allclose( gold_elem_boundary_coords, test_elem_boundary_coords))

#     def test_two_element_unit_domain(self):
#         gold_elem_boundary_coords = numby.array([0,0.5,1])
#         test_elem_boundary_coords = computeElemBoundaryCoords(xmin = 0, xmax = 1, num_elems = 2)
#         self.assertTrue( numpy.allclose( gold_elem_boundary_coords, test_elem_boundary_coords))
#         numpy.allclose( gold_elem_boundary_coords, test_elem_boundary_coords))

class Test_generateMesh( unittest.TestCase ):
    def test_make_1_linear_elem( self ):
        gold_node_coords = numpy.array( [ 0.0, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1 ] }
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, degree = [ 1 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    
    def test_make_1_quadratic_elem( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1, 2 ] }
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, degree = [ 2 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    
    def test_make_2_linear_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1 ], 1: [ 1, 2 ] }
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, degree = [ 1, 1 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    
    def test_make_2_quadratic_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1, 2 ], 1: [ 2, 3, 4 ] }
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, degree = [ 2, 2 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    
    def test_make_4_linear_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1 ], 1: [ 1, 2 ], 2: [ 2, 3 ], 3: [ 3, 4 ] }
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, degree = [ 1, 1, 1, 1 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    
    def test_make_4_quadratic_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1, 2 ], 1: [ 2, 3, 4 ], 2: [ 4, 5, 6 ], 3: [ 6, 7, 8 ] }
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 1.0, degree = [ 2, 2, 2, 2 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    
    def test_make_4_p_refine_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 1.0, 1.5, 2.0, (2.0 + 1.0/3.0), (2.0 + 2.0/3.0), 3.0, 3.25, 3.5, 3.75, 4.0 ] )
        gold_ien_array = { 0: [ 0, 1 ], 1: [ 1, 2, 3 ], 2: [ 3, 4, 5, 6 ], 3: [ 6, 7, 8, 9, 10 ] }
        node_coords, ien_array = generateMesh( xmin = 0.0, xmax = 4.0, degree = [ 1, 2, 3, 4 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )        