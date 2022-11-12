# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 20:45:47 2022

@author: brian
"""

import numpy


def getElementIdxContainingPoint(x, node_coords, ien_array):
    #loop through elements
    for i in range(0,len(ien_array)):
        #check if x is in element domain
        node_min = ien_array[i][0]
        node_max = ien_array[i][-1]
        if x >= node_min and x <= node_max:
            elem_idx = i
            return elem_idx

def getElementNodes(ien_array, elem_idx):
    nodes = []
    nodes = ien_array[elem_idx,:]
    return nodes

def getElementDomain(ien_array, node_coords, elem_idx):
    nodes = getElementNodes(ien_array, elem_idx)
    xmin = node_coords(nodes[0])
    xmax = node_coords(nodes[-1])
    elem_domain = [xmin, xmax]
    return elem_domain

def spatialToParamCoords(x, elem_domain):
    xmin = elem_domain[0]
    xmax = elem_domain[-1]
    x_param = ((2/(xmax-xmin))*x) + (-xmax-xmin)/(xmax-xmin)
    return x_param

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

def generateMeshwithDictionary(xmin, xmax, degree):
    num_elems = len(degree)
    elem_boundaries = numpy.linspace( xmin, xmax, num_elems + 1)
    ien_array = {}
    node_coords = []
    
    for elem_idx in range(0, num_elems):
        elem_xmin = elem_boundaries[elem_idx]
        elem_xmax = elem_boundaries[elem_idx]
        elem_nodes = numpy.linspace(elem_xmin, elem_xmax, degree[elem_idx] + 1)
        if elem_idx == 0:
            node_ids = numpy.arange(0, degree[elem_idx] + 1)
            node_coords.append(elem_nodes)
        else:
            start_node = ien_array[elem_idx -1][-1]
            node_ids = numpy.arange(start_node, start_node + degree[elem_idx])
            node_coords.append(elem_nodes[1:])
        ien_array[elem_idx] = list(node_ids)
    node_coords, ien_array
    return node_coords, ien_array
    
def generateMesh(xmin, xmax, degree):
    num_elems = len(degree)
    node_coords = []
    ien_array = []
    elem_bound = numpy.linspace(xmin, xmax, num_elems + 1)
    
    counter0 = int(0)
    elem_ien = []
    for i in range(degree[0] + 1):
        elem_ien.append(counter0)
        counter0 += int(1)
    ien_array.append(elem_ien)
    
    for j in range(num_elems):
        elem_xmin = elem_bound[j]
        elem_xmax = elem_bound[j+1]
        elem_nodes = numpy.linspace(elem_xmin, elem_xmax, degree[j] + 1)
        node_coords.append(elem_nodes)
        
        if j == 0:
            continue
        else: 
            elem_ien = []
            ien_array.append(elem_ien)
            counter1 = ien_array[-2][-1]
            for i in range(0, degree[j]+1):
                ien_array[-1].append(counter1)
                counter1 += int(1)
    return node_coords, ien_array

    