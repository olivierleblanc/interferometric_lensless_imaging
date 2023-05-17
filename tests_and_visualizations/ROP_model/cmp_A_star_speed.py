"""
    Copyright (c) 2021 Olivier Leblanc

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, and to permit
    persons to whom the Software is furnished to do so.
    However, anyone intending to sublicense and/or sell copies of the Software
    will require the official permission of the author.
    ----------------------------------------------------------------------------

    Author : Olivier Leblanc
    Date : 30/05/2022

    Code description :
    __________________
    Compares the speed of two implementations of the adjoint of the ROP operator.
    We observe that for Q=120 cores, the first implementation with 
    a_ij_outer is faster when M>1000.
    However, the duration is only around 1/10 thus one order of magnitude different.
"""

import matplotlib.pyplot as plt
import numpy as np
import time

import sys, os
def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
sys.path.append(os.path.join(updir(__file__,3),'utils'))

from interferometric_lensless_imaging import A_star, A_star2

nM = 30
# nQ = 1

Ms = np.logspace(2,4,nM).astype(int) # Number of measurements
# Qs = np.logspace(1.5,2, nQ).astype(int) # Number of cores
Q = 120

# time_ratio = np.zeros((len(Ms), len(Qs)))
time_ratio = np.zeros((len(Ms)))

for i, M in enumerate(Ms):
  # for j, Q in enumerate(Qs):
    print(i)
    # print(i,j)

    a_ij = (np.random.randn(M,Q)+1j*np.random.randn(M,Q))/np.sqrt(2)
    a_ij_outer = np.zeros((Q,Q,M), dtype=complex)
    for m in range(M):
        a_ij_outer[:,:,m] = np.outer(a_ij[m], a_ij[m].conj())

    y = np.abs(20*np.random.randn(M))

    tic = time.time()
    adj = A_star(y, a_ij_outer)
    toc = time.time()
    adj2 = A_star2(y, a_ij)
    tac=time.time()

    # time_ratio[i,j] = (toc-tic+np.finfo(float).eps)/(tac-toc+np.finfo(float).eps)
    time_ratio[i] = (toc-tic+np.finfo(float).eps)/(tac-toc+np.finfo(float).eps)

plt.figure()
plt.plot(Ms, time_ratio)
plt.xlabel('M')
plt.ylabel(r'$t/t_2$')
plt.yscale('log')
plt.ylim(1e-1, 1e1)
plt.show()