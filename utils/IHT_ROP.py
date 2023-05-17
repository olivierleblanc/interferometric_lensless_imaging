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
    Date : 08/12/2021

    Code description :
    __________________
    Implement the Iterative Hard Thresholding (IHT) algorithm for recovery from rank-one projections.

"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
from interferometric_lensless_imaging import H_r, A, A_star2
from functions import snr, is_hermitian


def IHT_ROP (Xn,y,a_ij,s,t,gamma=2,maxit=100, tol=1e-8, b_ij=None, diagless=True, is_hermitian=False, verbosity=0):
    """
    Apply Iterative Hard Thresholding algorithm (IHT) from rank-one projections (ROPs).
    The original paper can be found here: "https://arxiv.org/pdf/1810.11749".

    Args:
        Xn (2D array)       : Initial condition.
        y (1D array)        : ROPs of Xn.
        a_ij (2D array)     : measurement vectors used for the ROPs.
        gamma,s,t (floats)  : convergence parameters
        maxiter (int)       : maximum number of iterations
        tol (float)         : tolerance on the difference between two consecutive signals

    Returns:
        Xn (2D array)       : Reconstructed signal
    """
    diffs = np.zeros(maxit)
    diff=2*tol
    it=1

    A_op = lambda Xin: A(Xin, a_ij, b_ij=b_ij, diagless=diagless)
    At_op = lambda Xin: A_star2(Xin, a_ij, b_ij=b_ij, diagless=diagless)
    while (it<maxit and diff>tol):
        y_Ax = y-A_op(Xn)
        thresh = H_r(At_op( y_Ax/np.abs(y_Ax) ), t, is_hermitian)
        term = np.linalg.norm( thresh , 'fro')**2
        num = np.linalg.norm(y_Ax, 1)
        # den = max( \ # From the paper, but doesn't seem crucial.
        #     term , \
        #     1/(4*gamma**2) * ( (np.linalg.norm(A_op(thresh),1))**2) / term
        #     )
        # mu = num/den 
        mu = num / term

        Xn1 = H_r(Xn+mu*thresh, s, is_hermitian)
        diff = np.sum( np.abs( Xn1-Xn ) )
        diffs[it] = diff
        Xn = np.copy(Xn1)
        it+=1
        
        if (verbosity and it%50==0):
            print(it)

        # Add stopping criterion from paper

    if(it>=maxit):
        print('MAXIT reached: {} iterations'.format(maxit))
    else :
        print('Tolerance reached: |X_n1-X_n|={}'.format(tol))
    return Xn, diffs

#__________________________________________________________________________________________________________________________________________

# N = 60
# M = 800
# r = 3

# U = np.random.randn(N,r)+1j*np.random.randn(N,r)
# sig = np.diag(np.random.randn(r))
# # V_star = U.conj().T
# V_star = np.random.randn(r,N)+1j*np.random.randn(r,N)
# X = U@sig@V_star

# a_ij = (np.random.randn(M,N)+1j*np.random.randn(M,N))/np.sqrt(2) 
# b_ij = (np.random.randn(M,N)+1j*np.random.randn(M,N))/np.sqrt(2)

# "Choose noise level"
# y = A(X, a_ij, b_ij=b_ij, diagless=False)
# sig=1e-3
# y += sig*(np.random.randn(M)+1j*np.random.randn(M))/np.sqrt(2)

# x0 = np.zeros((N,N), dtype=complex)
# gamma=3
# s=r
# t=N # 2r as diff of two rank-r matrices is at most rank-2r or N to avoid thresholding
# X_iht, diffs = IHT_ROP (x0,y,a_ij,b_ij=b_ij,s=s,t=t,gamma=gamma,maxit=500, tol=1e-6, diagless=False, is_hermitian=is_hermitian(X), verbosity=1)


# print('SNR: {:.2f} dB'.format(snr(X, X_iht)) )
# rel_error_iht = np.linalg.norm(X-X_iht,ord='fro') / np.linalg.norm(X,ord='fro')
# print('Recovery with relative Frobenius error of {:.2e} by iterative hard thresholding'
#       .format(rel_error_iht))

# fig, axs = plt.subplots(1, 4, figsize=(18,5), gridspec_kw={'width_ratios':[1,1,0.05,1]})
# im0 = axs[0].imshow(np.abs(X), cmap='viridis')
# fig.colorbar(im0, cax=axs[2])
# im1 = axs[1].imshow(np.abs(X-X_iht), vmax=np.max(np.abs(X)))
# axs[0].set_title('GT')
# axs[1].set_title('Reconstruction Error')
# axs[3].plot(diffs[1:])
# axs[3].set_title(r'$|X_{n+1}-X_n|$')
# axs[3].set_yscale('log')
# fig.tight_layout()
# plt.show()