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
    Date : 16/08/2021

    Code description :
    __________________
    Generates random beta_ij's as stated in the notes, and observes its probability
    distribution to study if it's subgaussian. 
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
sys.path.append(os.path.join(updir(__file__,2),'utils'))

from functions import *

N = 256 # vector size
Q = 200 # Number of cores
M = 500 # Number of observations y

a_ij = np.exp(1j*2*np.pi*np.random.rand(M,Q))
beta = np.zeros((M,N), dtype=complex)

" Suggestion LJ "
for i in range(M):
    beta[i,np.random.permutation(np.arange(0,N))[:Q]] = a_ij[i,:]

beta2 = corr_circ(np.conj(beta)) # autocorrelations of beta
"Normalize each line"
norms = np.sqrt(np.sum( (np.abs(beta2))**2, axis=1))
beta2 = beta2/(np.kron(norms, np.ones(N)).reshape(M,N))

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.hist(np.angle(a_ij).reshape(-1), 100)
plt.title(r'arg $\alpha_{ij}$')

plt.subplot(122)
plt.hist(np.real(beta2).reshape(-1), 200)
# plt.xlim(0.6,0.8)
# plt.ylim(0,Q)
plt.title(r'$(\beta \otimes \beta)_{ij}$')
plt.show()