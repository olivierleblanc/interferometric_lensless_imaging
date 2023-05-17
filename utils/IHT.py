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
    Date : 19/11/2021

    Code description :
    __________________
    Implement the Iterative Hard Thresholding (ITH) algorithm.

"""
import numpy as np
from numpy.random import randn, randint
import matplotlib.pyplot as plt

def IHT (xn,y,s,A=None,At=None,maxit=100, tol=1e-8, nu=1, lamb=1, mode='L2', verbosity='low'):
    """
    Apply Iterative Hard Thresholding algorithm (IHT).

    Args:
        xn (1D array)             : Initial condition.
        y (1D array)              : ROPs of Xn.
        s (int)                   : Expected sparsity of x
        A (2D array or callable)  : Measurement matrix
        At (2D array or callable) : Adjoint of A  
        maxit (int)               : maximum number of iterations
        tol (float)               : tolerance on the difference between two consecutive signals
        nu (float or 1D array)    : gradient descent step
        lamb (float)              : L1-norm weight wrt fidelity term in the objective function
        mode (str)                : If 'L2', classical IHT, if 'L1' takes sign of the approximation error, thus minimizing the L1-norm of the error.
                                    See "An IHT Algorithm for Sparse Recovery From Subexponential Measurements" for details.
        verbosity (str)           : Print diff at some iterations if 'high', doesn't print otherwise.

    Returns:
        xn (1D array)             : Reconstructed signal
        diffs (1D array)          : Absolute difference between each iterate
    """
    if A is None:
        A2 = lambda x: x
    else:
        if callable(A):
            A2 = A
        else:
            # Transform matrix form to operator form.
            A2 = lambda x: A.dot(x)

    if At is None:
        if A is None:
            A2t = lambda x: x
        elif callable(A):
            A2t = A
        else:
            A2t = lambda x: A.conj().T.dot(x)
    else:
        if callable(At):
            A2t = At
        else:
            A2t = lambda x: At.dot(x)

    objective = np.zeros(maxit)
    diffs = np.zeros(maxit)
    diff=2*tol
    nu_it = nu
    if (not hasattr(nu, "__len__")):
        nu = np.repeat(nu, maxit)
    it=0

    while (it<maxit and diff>tol):
        y_Ax = y-A2(xn)
        if (mode=='L1'):
            y_Ax = np.sign(y_Ax)
        nu_it = nu[it]
        tmp = xn+nu_it*A2t(y_Ax)
        abstmp = np.abs(tmp)
        ord_tmp = np.sort(abstmp.reshape(-1))
        thres = ord_tmp[-s]
        xn1 = tmp* (abstmp>=thres)
        diff = np.sum( np.abs( xn1-xn ) )
        diffs[it] = diff
        objective[it] = 0.5*np.linalg.norm(A2(xn1)-y,2)**2 + lamb*np.linalg.norm(xn1,1)

        if (it%50==0 and verbosity=='high'):
            print(it, diff)
        xn = np.copy(xn1)
        it+=1
    if(it>=maxit):
        print('MAXIT reached: {} iterations'.format(maxit))
    else :
        print('Tolerance reached at iteration {}: |X_n1-X_n|={}'.format(it-1,tol))
        diffs = diffs[:it]
        objective = objective[:it]
    return xn, diffs, objective

#__________________________________________________________________________________________________________________________________________

# N = 2000
# M = 400
# s = 20

# inds = randint(N, size=s)
# x = np.zeros(N, dtype=complex)
# x[inds]=randn(s)+1j*randn(s)

# A = (randn(M,N)+1j*randn(M,N))/np.sqrt(2*M)

# "Choose noise level"
# sig=1e-2
# y = A@x + sig*(randn(M)+1j*randn(M))/np.sqrt(2)

# x0 = np.zeros(N, dtype=complex)
# test, diffs = IHT(x0, y, s=s, A=A, maxit=50, mode='L2', nu=1e0)

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
# axs[0].plot(np.abs(x), 'o', label='original')
# axs[0].plot(np.abs(test),'xr', label='reconstructed')
# axs[0].legend()
# axs[1].plot(diffs)
# axs[1].set_yscale('log')
# axs[1].set_ylabel(r'$|x_{n+1}-x_n|$')
# plt.show()