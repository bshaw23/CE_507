# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:21:55 2022

@author: brian
"""

import numpy
import quadrature as quad
import Basis as basis
import mesh

def change_of_coords(domain_old, domain_new, x_old):
    a = domain_old[0]
    b = domain_old[1]
    c = domain_new[0]
    d = domain_new[1]
    x_new = ((d-c)/(b-a))*x_old + ((b*c-a*d)/(b-a))
    return x_new

def jacobian(domain_old, domain_new):
    a = domain_old[0]
    b = domain_old[1]
    c = domain_new[0]
    d = domain_new[1]
    jacobian = (d-c)/(b-a)
    return jacobian

def assembleGramMatrixBernstien(domain, degree, n):
    qp, w = quad.computeGaussLegendreQuadrature(n)
    M = numpy.zeros([degree+1, degree+1])
    jacob = jacobian([-1,1], domain)
    for i in range(degree+1):
        for j in range(degree+1):
            for q in range(len(qp)):
                N_A = basis.evalBernsteinBasis1D(qp[q], degree, i)
                N_B = basis.evalBernsteinBasis1D(qp[q], degree, j)
                M[i,j] += N_A*N_B*w[q]*jacob
    return M

def assembleGramMatrixLagrange(domain, degree, n):
    qp, w = quad.computeGaussLegendreQuadrature(n)
    M = numpy.zeros([degree+1, degree+1])
    jacob = jacobian([-1,1], domain)
    for i in range(degree+1):
        for j in range(degree+1):
            for q in range(len(qp)):
                N_A = basis.evalLagrangeBasis1D(qp[q], degree, i)
                N_B = basis.evalLagrangeBasis1D(qp[q], degree, j)
                M[i,j] += N_A*N_B*w[q]*jacob
    return M

def assembleGramMatrixLegendre(domain, degree, n):
    qp, w = quad.computeGaussLegendreQuadrature(n)
    M = numpy.zeros([degree+1, degree+1])
    jacob = jacobian([-1,1], domain)
    for i in range(degree+1):
        for j in range(degree+1):
            for q in range(len(qp)):
                N_A = basis.evalLegendreBasis1D(i, qp[q])
                N_B = basis.evalLegendreBasis1D(j, qp[q])
                M[i,j] += N_A*N_B*w[q]*jacob
    return M

def assembleForceVectorBernstien(target_fun, domain, degree, n):
    domain_new = domain
    domain_old = [-1,1]
    jacob = jacobian(domain_old,domain_new)
    qp, w = quad.computeGaussLegendreQuadrature(n)
    F = numpy.zeros((degree+1))
    
    for i in range(degree+1):
        for q in range(degree+1):
            N_A = basis.evalBernsteinBasis1D(qp[q], degree, i)
            F[i] += N_A * target_fun(change_of_coords(domain_old, domain_new, qp[q])) * w[q]*jacob
    return F 

def assembleForceVectorLagrange(target_fun, domain, degree,n):
    domain_new = domain
    domain_old = [-1,1]
    jacob = jacobian(domain_old,domain_new)
    qp, w = quad.computeGaussLegendreQuadrature(n)
    F = numpy.zeros((degree+1))
    
    for i in range(degree+1):
        for q in range(degree+1):
            N_A = basis.evalLagrangeBasis1D(qp[q], degree, i)
            F[i] += N_A * target_fun(change_of_coords(domain_old, domain_new, qp[q])) * w[q] * jacob
    return F

def assembleForceVectorLegendre(target_fun, domain, degree, n):
    domain_new = domain
    domain_old = [-1,1]
    jacob = jacobian(domain_old,domain_new)
    qp, w = quad.computeGaussLegendreQuadrature(n)
    F = numpy.zeros((degree+1))
    
    for i in range(degree+1):
        for q in range(degree+1):
            N_A = basis.evalLegendreBasis1D(i, qp[q])
            F[i] += N_A * target_fun(change_of_coords(domain_old, domain_new, qp[q])) * w[q] * jacob
    return F
