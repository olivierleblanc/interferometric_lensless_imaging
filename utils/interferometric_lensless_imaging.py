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
    Date : 27/05/2022

    Code description :
    __________________
    Contains the forward and adjoint operators for the interferometric lensless imaging model.

"""
import types
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2, ifft2
import spgl1
from scipy.sparse.linalg import LinearOperator
import pywt

# Below, this way of refering to the right path is more portable (valid on Windows, Mac, and Linux)
import sys, os

from numpy.lib.type_check import iscomplex
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'utils')) 

from utils_wavelet import *
from graphics import set_plot_params, subplot_axs, labels
from functions import is_complex, is_hermitian

#________________________________________________________________________________
#                                   Code
#________________________________________________________________________________

def adjoint_test(A,A_star,A_shape, u=None, v=None):
    """
    Check if operator A_star is well the adjoint of operator A.
    """
    if (v is None):
        if len(A_shape[0]) == 1: 
            v = np.random.randn( A_shape[0][0] )
        elif len(A_shape[0]) == 2:
            v = np.random.randn(A_shape[0][0], A_shape[0][1])
        else:
            raise ValueError('invalid dimension')
    if (u is None):
        if len(A_shape[1]) == 1: 
            u = np.random.randn( A_shape[1][0] )
        elif len(A_shape[1]) == 2:
            u = np.random.randn(A_shape[1][0], A_shape[1][1])
        else:
            raise ValueError('invalid dimension')

    if isinstance(A, types.FunctionType):
        A_op = A
        At_op = A_star
    else:
        A_op = lambda x: A@x
        At_op = lambda y: A_star@y

    score = np.abs(np.vdot(u, A_op(v)) - np.vdot(At_op(u), v))/(np.linalg.norm(u)*np.linalg.norm(A_op(v)))

    if score < 1e-12:
        print(f"A and At are adjoint!")    
    else:
        print(f"Warning: A and At are not adjoint, |<u,Av>-<A*u,v>|={score:2.2e}||u|| ||Av||")  

def rmdiag(X):
    """Remove the diagonal of a matrix X"""
    return X - np.diag(np.diag(X))

def diago(X):
    """Return the diagonal of a matrix X"""
    return np.diag(X)

def H_r(X, r, is_hermitian=False):
    """
    Computes projection on matrices of rank "r"
    
    Args:
        X (2D array)        : Signal to be projected.
        r (int)             : target rank
        is_hermitian (bool) : True if X is hermitian 

    Returns:
        projection of X onto the set of rank-r matrices.
    """
    U, s, V = np.linalg.svd(X, hermitian=is_hermitian)
    return U[:,:r] @ np.diag(s[:r]) @ V[:r,:]


def eval_nu(init, A, A_star=None, nb_iter = 10):
    """Estimate the square norm of the operator B (i.e. ||B||^2) thanks to the power method.
        Useful to bound the norm of B, i.e. give |B(x)|^2 \leq \nu |x|^2`.
    """ 
    if isinstance(A, types.FunctionType):
        if (A_star is None):
            raise ValueError("A_star must be provided")
        A_op = A
        At_op = A_star
    else:
        if (A_star is None):
            A_star = A.T
        A_op = lambda x: A@x
        At_op = lambda y: A_star@y

    u = init
    for k in range(nb_iter):
        u = u/np.linalg.norm(u) # Normalize current matrix
        u = At_op(A_op(u))            
    return np.linalg.norm(u)


def LE_cores (diam_endo, nCores, sources_shape = 'fermat'):
    """Outputs the cores positions at the output of the lensless endoscope (LE).

    Args:
        diam_endo (float)  : diameter of the LE.
        nCores (int)       : number of cores in the LE.
        sources_shape (str): spatial arrangement of the cores, 'rect', 'radial' or 'fermat'.

    Returns:
        pos_sources (2D array): [xs, ys] 2D positions.

    """
    if (sources_shape =='rect'):
        xs = np.linspace(-diam_endo/2/np.sqrt(2), diam_endo/2/np.sqrt(2), round(np.sqrt(nCores)))
        ys = np.linspace(-diam_endo/2/np.sqrt(2), diam_endo/2/np.sqrt(2), round(np.sqrt(nCores)))
        xmesh,ymesh = np.meshgrid(xs,ys)
        pos_sources = np.array([xmesh.reshape(-1), ymesh.reshape(-1)])
    elif (sources_shape == 'radial'): 
        radii = np.linspace(0,diam_endo/2,round((nCores)**(1/3)))
        theta = np.linspace(0,2*np.pi, 2*round((nCores)**(1/3)))
        allrad, alltheta = np.meshgrid(radii, theta)
        pos_x = allrad*np.cos(alltheta)
        pos_y = allrad*np.sin(alltheta)
        pos_sources = np.array([pos_x.reshape(-1), pos_y.reshape(-1)])
    elif (sources_shape == 'fermat'):
        c = 0.9*diam_endo/(2*np.sqrt(nCores))
        n_array = np.arange(nCores)
        rho = c*np.sqrt(n_array)
        phi = n_array*np.pi*(3-np.sqrt(5))
        pos_x = rho*np.cos(phi)
        pos_y = rho*np.sin(phi)
        pos_sources = np.array([pos_x, pos_y])
    
    return pos_sources

#_____________________________________________________________________________________
#                   Forward and inverse problem operators
#_____________________________________________________________________________________
def T(U, axes=None):
    """
    1D or 2D Fourier transform  
    """
    if (axes is None):
        if (U.ndim==1):
            return np.fft.fft(U)/np.sqrt(len(U))
        elif (U.ndim==2):
            return np.fft.fft2(U)/U.shape[0]
        else:
            raise ValueError("Input must be 1D or 2D")
    else:
        if (not hasattr(axes, '__iter__')):
            return np.fft.fft(U, axis=axes)/np.sqrt(U.shape[axes])
        elif (len(axes)==2):
            return np.fft.fft2(U,axes=axes)/U.shape[axes[0]]
        else:
            raise ValueError("Input must be 1D or 2D")

def T_star(Uf, axes=None):
    """
    Inverse Fourier transform  
    """
    if (axes is None):
        if (Uf.ndim==1):
            return np.fft.ifft(Uf)*np.sqrt(len(Uf))
        elif (Uf.ndim==2):
            return np.fft.ifft2(Uf)*Uf.shape[0]
        else:
            raise ValueError("Input must be 1D or 2D")
    else:
        if (not hasattr(axes, '__iter__')):
            return np.fft.ifft(Uf, axis=axes)/np.sqrt(Uf.shape[axes])
        elif (len(axes)==2):
            return np.fft.ifft2(Uf,axes=axes)/Uf.shape[axes[0]]
        else:
            raise ValueError("Input must be 1D or 2D")

def ind_multiplicity_1D(Om):
    """Return the multiplicity of Om.
    Args:
    Om (2D array): indices
    Returns:
    W (2D array): multiplicity of each Om
    """
    Om_flat = Om.flatten()
    _, inv_ind, W = np.unique(Om_flat, axis=0, return_inverse=True, return_counts=True)
    W = W[inv_ind]
    W = np.reshape(W,Om.shape)
    return W

def ind_multiplicity(Om_x, Om_y):
    """Return the multiplicity of Om_x, Om_y.
    Args:
    Om_x, Om_y (2D arrays): x and y components of (meshgridded) indices
    Returns:
    W (2D arrays): multiplicity of each (Om_x,Om_y) pair
    """
    Om_x_flat = Om_x.flatten()
    Om_y_flat = Om_y.flatten()
    pairs = np.array([[Om_y_flat[i],Om_x_flat[i]] for i in range(Om_x_flat.shape[0])])
    _, inv_ind, W = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    W = W[inv_ind]
    W = np.reshape(W,Om_x.shape)
    return W

def S_Om_1D (x, Om, mult_mat=None):
    """Subsample 1D signal 'x' at indices 'Om'.
       The multiplicity of each index is corrected.

    Args:
        x (1D array) : Input signal.
        Om (2D array): The indices

    Returns:
        S_om_x (2D array): x[Om].

    """
    S_Om_x = x[Om] 
    
    if (mult_mat is not None):
        S_Om_x = x[Om]/np.sqrt(mult_mat)

    # "Avoid aliasing"
    # ind = np.abs(Om)>=len(x)/2 
    # S_Om_x[ind] = 0

    return S_Om_x

def S_Om_star_1D (U, Om, objshape, mult_mat=None):
    """Adjoint of S_Om. Fill a 1D array at indices 'Om' with 'u'.
       Note: regarding the notes, S_Om_star not compact must consider the multiplicities,
             they are thus added to keep the non compact format. 
       The 'diag(w)' term is finally corrected with diag(1/sqrt(w)) in both S_Om and S_Om_star

    Args:
        U (2D array)  : Input signal.

    Returns:
        S_om_star_U (1D array).

    """
    S_Om_star_U = np.zeros(objshape, dtype=complex)

    if (mult_mat is None):
        np.add.at(S_Om_star_U, Om, U) 
    else :
        np.add.at(S_Om_star_U, Om, U/np.sqrt(mult_mat)) 
            
    return S_Om_star_U

def S_Om (X, Om_x, Om_y, mult_mat=None):
    """Subsample 2D signal 'X' at indices 'Om_x' and 'Om_y'.
       The multiplicity of each index is corrected.

    Args:
        X (2D array)        : Input signal.
        Om_x (1D array)     : core position pairwise difference along x-axis
        Om_y (1D array)     : core position pairwise difference along y-axis
        mult_mat (2D array) : matrix containing the multiplicity of each Fourier coeff

    Returns:
        S_om_X (2D array): X[Om_x,Om_y].

    """
    S_Om_X = X[Om_y,Om_x] 
    
    if (mult_mat is not None):
        S_Om_X = X[Om_y,Om_x]/np.sqrt(mult_mat)
      
    return S_Om_X

def S_Om_star (U, Om_x, Om_y, objshape, mult_mat=None):
    """Adjoint of S_Om. Fill a 2D array at indices 'Om_x' and 'Om_y' with 'U'.
       Note: regarding the notes, S_Om_star not compact must consider the multiplicities,
             they are thus added to keep the non compact format. 
       The 'diag(w)' term is finally corrected with diag(1/sqrt(w)) in both S_Om and S_Om_star

    Args:
        U (2D array)  : Input signal.

    Returns:
        S_om_star_U (2D array).

    """
    S_Om_star_U = np.zeros(objshape, dtype=complex)

    if (mult_mat is None):
        np.add.at(S_Om_star_U, (Om_y, Om_x), U) 
    else :
        np.add.at(S_Om_star_U, (Om_y, Om_x), U/np.sqrt(mult_mat)) 
            
    return S_Om_star_U


# standard ROP, A and adjoint A_star
def A(X, a_ij, b_ij=None, diagless=True):
    """Generate the M ROPs of X by vectors a_i.
       If diagless is True, the ROPs are diagonal less.

    Args:
        X (2D array)   : matrix onto apply the ROPs.
        a_ij           : matrix whose lines are the vectors a_i.
        b_ij           : matrix whose lines are the vectors b_i (optional).
        diagless (bool): True (False)-> Neglect (Consider) diagonal of X for the ROPs 

    Returns:
        y (1D array)   : all ROPs

    """
    M = a_ij.shape[0]

    if (diagless):
        X = rmdiag(X)
    if (b_ij is None):
        y = np.sum( (a_ij.conj()*(a_ij@X.T)), axis=1)
    else:
        assert a_ij.shape==b_ij.shape, "a_ij and b_ij should contain the same number of elements"
        y = np.sum( (a_ij.conj()*(b_ij@X.T)), axis=1) # Written properly this is equivalent
    return y/np.sqrt(M)


def A_star(y, a_ij_outer, diagless=True):
    """Adjoint of the ROPs.
       If diagless is True, the result is diagonal less.

    Args:
        y (1D array)   : all ROPs.
        a_ij           : matrix whose lines are the vectors a_i.
        diagless (bool): True (False)-> Neglect (Consider) diagonal of X for the ROPs 

    Returns:
        z (2D array)   : 

    """
    _, _, M = a_ij_outer.shape
    z = a_ij_outer@y

    if (diagless):
        z = rmdiag(z)
    return z/np.sqrt(M)

def A_star2(y, a_ij, b_ij=None, diagless=True):
    """Adjoint of the ROPs.
       If diagless is True, the result is diagonal less.

    Args:
        y (1D array)   : all ROPs.
        a_ij           : matrix whose lines are the vectors a_i.
        b_ij           : matrix whose lines are the vectors b_i (optional).
        diagless (bool): True (False)-> Neglect (Consider) diagonal of X for the ROPs 

    Returns:
        z (2D array)   : 

    """
    M = a_ij.shape[0]

    if (b_ij is None):
        z = a_ij.T @ np.diag(y) @ a_ij.conj()
    else:
        assert a_ij.shape==b_ij.shape, "a_ij and b_ij should contain the same number of elements"
        z = a_ij.T @ np.diag(y) @ b_ij.conj()

    if (diagless):
        z = rmdiag(z)
    return z/np.sqrt(M)

#_____________________________________________________________________________________
#                   LinearOperator instances for SPGL1
#_____________________________________________________________________________________

class ROP_model(LinearOperator):
    '''
    Class inheriting LinearOperator from scipy.sparse.linalg
    Implements the forward and backward observation operators of the ROP model.
        Om: array_like - core position pairwise difference.
        a_ij: array_like - matrix whose lines are the vectors a_i.
        b_ij: array_like - matrix whose lines are the vectors b_i (optional).
        N: int - object size.
        wt (optional): str - the chosen wavelet.
        mult_mat: array_like - contains the multiplicity of each frequency in the interferometric matrix.
    '''
    def __init__(self, Om, a_ij, N, b_ij=None, wt=None, mult_mat=None, level=None, diagless=False):
        self.M, self.Q = a_ij.shape
        self.N = N
        self.shape = (self.M,self.N)
        self.dtype = np.complex128
        self.diagless = diagless
        self.Om = Om
        self.a_ij = a_ij
        self.b_ij = b_ij
        self.a_ij_outer = np.zeros((self.Q,self.Q,self.M), dtype=complex)
        for m in range(self.M):
            self.a_ij_outer[:,:,m] = np.outer(a_ij[m], a_ij[m].conj())
        self.wt = wt
        self.level = level
        if (self.wt is not None):
            self.template = pywt.wavedec2(np.zeros((self.N,self.N)), wt, mode='periodization', level=self.level)
        self.mult_mat = mult_mat
        
    def wav(self, x):
        return arrayList2vec(pywt.wavedec(x, self.wt, mode='periodization', level=self.level))

    def wavT(self, x):
        return pywt.waverec(vec2arrayList(x, self.template), self.wt, mode='periodization')

    def _matvec(self, x):
        if (self.wt is None):
            sparse_obj = x
        else :
            sparse_obj = self.wavT(x)
        return (A(S_Om_1D(T(sparse_obj), self.Om, mult_mat=self.mult_mat), self.a_ij, b_ij=self.b_ij, diagless=self.diagless)).real
    
    def _rmatvec(self, y):
        # inter = T_star(S_Om_star_1D(A_star(y, self.a_ij_outer, diagless=diagless), self.Om, self.N, mult_mat=self.mult_mat ) )
        inter = T_star(S_Om_star_1D(A_star2(y, self.a_ij, b_ij=self.b_ij, diagless=self.diagless), self.Om, self.N, mult_mat=self.mult_mat ) )
        if (self.wt is None):
            return inter.real
        else :
            return (self.wav(inter)).real


class ROP_model2(LinearOperator):
    '''
    Class inheriting LinearOperator from scipy.sparse.linalg
    Implements the forward and backward observation operators of the ROP LE model including an optional wavelet operator in 2D.
        Om_x, Om_y: array_like - core position pairwise difference along x- and y-axis, respectively.
        a_ij: array_like - matrix whose lines are the vectors a_i.
        b_ij: array_like - matrix whose lines are the vectors b_i (optional).
        N: int - object size.
        wt (optional): str - the chosen wavelet.
        mult_mat: array_like - contains the multiplicity of each frequency in the interferometric matrix.
    '''
    def __init__(self, Om_x, Om_y, a_ij, N, b_ij=None, wt=None, mult_mat=None, level=None, diagless=False):
        self.M, self.Q = a_ij.shape
        self.N = N
        self.shape = (self.M,self.N**2)
        self.dtype = np.complex128
        self.diagless = diagless
        self.Om_x = Om_x
        self.Om_y = Om_y
        self.a_ij = a_ij
        self.b_ij = b_ij
        self.a_ij_outer = np.zeros((self.Q,self.Q,self.M), dtype=complex)
        for m in range(self.M):
            self.a_ij_outer[:,:,m] = np.outer(a_ij[m], a_ij[m].conj())
        self.wt = wt
        self.level = level
        if (self.wt is not None):
            self.template = pywt.wavedec2(np.zeros((self.N,self.N)), wt, mode='periodization', level=self.level)
        self.mult_mat = mult_mat

    def wav2(self, x):
        return arrayList2vec(pywt.wavedec2(x, self.wt, mode='periodization', level=self.level))

    def wavT2(self, x):
        return pywt.waverec2(vec2arrayList(x, self.template), self.wt, mode='periodization')

    def _matvec(self, x):
        if (self.wt is None):
            x = x.reshape((self.N,self.N))
            sparse_obj = x
        else :
            sparse_obj = self.wavT2(x)
        return (A(S_Om(T(sparse_obj), self.Om_x, self.Om_y, mult_mat=self.mult_mat), self.a_ij, b_ij=self.b_ij, diagless=self.diagless)).real 
    
    def _rmatvec(self, y):
        # inter = T_star(S_Om_star(A_star(y, self.a_ij_outer, diagless=diagless), self.Om_x, self.Om_y, (self.N, self.N), mult_mat=self.mult_mat) )
        inter = T_star(S_Om_star(A_star2(y, self.a_ij, b_ij=self.b_ij, diagless=self.diagless), self.Om_x, self.Om_y, (self.N, self.N), mult_mat=self.mult_mat) )
        if (self.wt is None):
            return inter.real
        else :
            return (self.wav2(inter)).real


class Autocorr_model(LinearOperator):
    '''
    Class inheriting LinearOperator from scipy.sparse.linalg
    Implements the forward and backward observation operators of the autocorrelation LE model including an optional wavelet operator.
        beta2: array_like - The autocorrelations of each beta arranged in rows.
        wt (optional): str - the chosen wavelet
    '''
    def __init__(self, beta2, wt=None, level=None):
        self.M, self.N = beta2.shape
        self.shape = beta2.shape
        self.dtype = np.complex128
        self.beta2 = beta2
        self.wt = wt
        self.level = level
        if (self.wt is not None):
            self.template = pywt.wavedec(np.zeros(self.N), wt, mode='periodization', level=self.level)

    def wav(self, x):
        return arrayList2vec(pywt.wavedec(x, self.wt, mode='periodization', level=self.level))

    def wavT(self, x):
        return pywt.waverec(vec2arrayList(x, self.template), self.wt, mode='periodization')

    def _matvec(self, x):
        if (self.wt is None):
            sparse_obj = x
        else :
            sparse_obj = self.wavT(x)
        return (self.beta2@T(sparse_obj)).real/np.sqrt(self.M)
    
    def _rmatvec(self, y):
        inter = T_star(self.beta2.conj().T@y)
        if (self.wt is None):
            return inter.real/np.sqrt(self.M)
        else :
            return (self.wav(inter)).real/np.sqrt(self.M)

class Autocorr_model2(LinearOperator):
    '''
    Class inheriting LinearOperator from scipy.sparse.linalg
    Implements the forward and backward observation operators of the autocorrelation LE model including an optional wavelet operator in 2D.
        B2: array_like - The autocorrelations of each B arranged in rows.
        wt (optional): str - the chosen wavelet
    '''
    def __init__(self, B2, wt=None, level=None):
        self.N,_, self.M = B2.shape
        self.shape = (self.M,self.N**2)
        self.dtype = np.complex128
        self.B2 = B2
        self.wt = wt
        self.level = level
        if (self.wt is not None):
            self.template = pywt.wavedec2(np.zeros((self.N,self.N)), wt, level=self.level, mode='periodization')

    def wav2(self, x):
        return arrayList2vec(pywt.wavedec2(x, self.wt, level=self.level, mode='periodization'))

    def wavT2(self, x):
        return pywt.waverec2(vec2arrayList(x, self.template), self.wt, mode='periodization')
        
    def _matvec(self, X):
        if (self.wt is None):
            X = X.reshape((self.N,self.N))
            sparse_obj = X
        else :
            sparse_obj = self.wavT2(X)
        return np.real(np.tensordot(T(sparse_obj), self.B2)) /np.sqrt(self.M)
    
    def _rmatvec(self, y):
        inter = T_star(np.conj(self.B2@y))
        if (self.wt is None):
            return inter.real/np.sqrt(self.M)
        else :
            return (self.wav2(inter)).real/np.sqrt(self.M)

class Op2Real(LinearOperator):
    """
    A function to turn complex linear operators (applied to real vectors) into real linear operators with two times as many rows.
    """
    def __init__(self, A):
        self.A = A
        m, n = A.shape
        self.shape = [2*m, n]
        self.dtype = np.float64
        
    def _matvec(self, x):
        return np.concatenate(((self.A @ x).real,
                               (self.A @ x).imag))
    
    def _rmatvec(self, y):
        m = self.A.shape[0]
        return (self.A.T @ y[0:m]).real + (self.A.T @ y[m:]).imag