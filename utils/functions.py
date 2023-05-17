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
    Date : 05/05/2022

    Code description :
    __________________
    Contains useful functions for data processing. 

"""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from PIL import Image

"SNR"
snr = lambda ref, rec : 20*np.log10(np.linalg.norm(ref)/np.linalg.norm(rec-ref))

def PSNR(truth, approximation):
    MSE  = ((truth-approximation)**2).mean()
    PSNR = 10*np.log10(1**2/MSE)
    return PSNR

def is_complex(sig):
    """Check if signal X is complex, return 1 if yes, 0 otherwise. """
    return np.sum(np.abs(np.imag(sig))>1e-8)>0
    # return isinstance(sig, complex)

def is_hermitian(X):
    """Check if matrix X is hermitian, return 1 if yes, 0 otherwise. """
    return np.mean(np.abs(X-X.T.conj()))<1e-10  

def gaussian(x, mu, sig, mode='amplitude'):
    """Return a 1D gaussian curve with mean ``mu`` and standard deviation ``sig``.
    If ``mode`` is ``amplitude``, max value is 1."""
    curve = np.exp( -(x-mu)**2/(2*sig**2))
    if (mode=='energy'):
        curve /= np.sqrt(2*np.pi*sig**2)
    return curve

def gaussian_dsig(x, mu, sig, mode='amplitude'):
    """Return the first derivative of a 1D gaussian curve with mean ``mu`` and 
    standard deviation ``sig`` with respect to its std.
    If ``mode`` is ``amplitude``, max value is 1."""
    deriv = (x-mu)**2/(sig**3)*gaussian(x,mu,sig,mode=mode)
    if (mode=='energy'):
        deriv -= 1/sig*gaussian(x,mu,sig,mode=mode)
    return deriv

def gaussian2(xx, yy, mu, sig, mode='amplitude'):
    """Return a 2D gaussian signal with mean ``mu`` and standard deviation ``sig``.
    If ``mode`` is ``amplitude``, max value is 1."""
    signal = np.exp( -((xx-mu[0])**2+(yy-mu[1])**2)/(2*sig**2))
    if (mode=='energy'):
        signal /= np.sqrt(2*np.pi*sig**2)
    return signal

def gaussian_dsig2(xx, yy, mu, sig, mode='amplitude'):
    """Return the first derivative of a 2D gaussian signal with mean ``mu`` and 
    standard deviation ``sig`` with respect to its std.
    If ``mode`` is ``amplitude``, max value is 1."""
    deriv = ((xx-mu[0])**2+(yy-mu[1])**2)/(sig**3)*gaussian2(xx, yy, mu,sig)
    if (mode=='energy'):
        deriv -= 1/sig*gaussian2(xx,yy,mu,sig,mode=mode)
    return deriv

def gaussian_kernel(sigma,image):
    """
    Returns an image-shaped gaussian kernel (point spread function) with scale sigma.
    """
    nx, ny = image.shape
    indx = np.linspace(-nx/2,nx/2,nx)
    indy = np.linspace(-ny/2, ny/2, ny)
    [X,Y] = np.meshgrid(indx,indy)
    h = np.exp( -(X**2+Y**2)/(2.0*(sigma)**2) )
    # h /= h.sum() # Normalize
    return h

def resize_and_fix_origin(kernel, size):
    """Pads a kernel to reach shape `size`, and shift it in order to cancel phase.
    This is based on the assumption that the kernel is centered in image space.
    """
    pad0, pad1 = size[0]-kernel.shape[0], size[1]-kernel.shape[1]
    # shift less if kernel is even, to start with 2 central items
    shift0, shift1 = (kernel.shape[0]-1)//2, (kernel.shape[1]-1)//2

    kernel = np.pad(kernel, ((0,pad0), (0,pad1)))
    kernel = np.roll(kernel, (-shift0, -shift1), axis=(0,1))
    return kernel

def zeropad(arr, factor):
    "Pad an array with zeros all around to multiply the array shape by factor."
    return np.pad(arr, tuple([(int(factor*val/2),)*2 for val in arr.shape]) )

def crop(arr, factor):
    "Crop an array around the center to divide the array shape by factor."
    s = arr.shape
    d = arr.ndim
    tmp = np.copy(arr)
    for i,_ in enumerate(np.arange(d)):
        tmp = tmp[int(s[i]/2) - int(s[i]/(2*(factor+1))):-int(s[i]/2) + int(s[i]/(2*(factor+1)))]
        tmp = np.transpose(tmp)
    return tmp

def fast_convolution(image, kernel):
    """
    Convolution through FFT
    """
    bfft = fft2(image)
    kfft = fft2(resize_and_fix_origin(kernel, image.shape))
    result = ifft2(kfft*bfft).real
    
    return result

def convolve(image, kernel) :
    """
    Compute the convolution through Fourier transform
    """
    imfft = fft2(image)
    kfft = fft2(resize_and_fix_origin(kernel, image.shape))
    return ifft2(kfft*imfft).real

def conv_circ(a, b=None):
    '''
    Circular convolution
        a: array_like
        b: array_like, optional
        a and b must have same length in last dimension
    '''
    if (b is not None):
        if (b.ndim == 1):    
            assert (len(a)==len(b)), "The input must have the same length."
        else:    
            assert (a.shape[1]==b.shape[1]), "The input must have the same length."
        tmp = np.fft.ifft( np.fft.fft(a)*np.fft.fft(np.conj(b)) )
        if (is_complex(a)+is_complex(b)==0):
            tmp = np.real(tmp)
        return tmp
    else:
        Fa = np.fft.fft(a)
        if (a.ndim == 1):  
            Fa2 = Fa[::-1]
        else: 
            Fa2 = Fa[:,::-1]
        Fa2 = np.roll(Fa2,1,axis=-1)
        tmp = np.fft.ifft( Fa*np.conj(Fa2))
        if (~is_complex(a)):
            tmp = np.real(tmp)
        return tmp

def corr_circ(a, b=None):
    '''
    Circular correlation
        a: array_like
        b: array_like, optional
        a and b must have same length in last dimension
    '''
    if (b is not None):
        if (b.ndim == 1):    
            assert (len(a)==len(b)), "The input must have the same length."
        else:    
            assert (a.shape[1]==b.shape[1]), "The input must have the same length."

        tmp = np.fft.ifft( np.fft.fft(a)*np.fft.fft(np.conj(b[::-1])))
        if (is_complex(a)+is_complex(b)==0):
            tmp = np.real(tmp)
        return tmp
    else:
        Fa = np.fft.fft(a)
        tmp = np.fft.ifft( np.abs(Fa)**2)
        if (~is_complex(a)):
            tmp = np.real(tmp)
        return tmp


def conv_circ2(a, b=None):
    '''
    Circular 2D convolution 
        a: array_like
        b: array_like, optional
        a and b must have same shape in 2D first dimensions
    '''
    if (b is not None):
        if (b.ndim == 2):    
            assert (a.shape == b.shape), "The input must have the same shape."
        else:    
            assert (a.shape[:1]==b.shape[:1]), "The input must have the same shape."
        tmp = np.fft.ifft2( np.fft.fft2(a, axes=(0,1,))*np.fft.fft2(np.conj(b), axes=(0,1,)), axes=(0,1,) )
        if (is_complex(a)+is_complex(b)==0):
            tmp = np.real(tmp)
        return tmp
    else:
        Fa = np.fft.fft2(a, axes=(0,1,)) # Taken on the two last axes.
        if (a.ndim == 2):  
            Fa2 = Fa[::-1,::-1]
        elif (a.ndim == 3): 
            Fa2 = Fa[::-1,::-1,:]
        else:
            raise ValueError('Input is not 2D or 3D.')
        Fa2 = np.roll(Fa2, 1, axis=0)
        Fa2 = np.roll(Fa2, 1, axis=1)
        
        tmp = np.fft.ifft2( Fa*np.conj(Fa2), axes=(0,1,))
        if (~is_complex(a)):
            tmp = np.real(tmp)
        return tmp

def corr_circ2(a, b=None):
    '''
    Circular 2D correlation
        a: array_like
        b: array_like, optional
        a and b must have same shape in 2D first dimensions
    '''
    if (b is not None):
        if (b.ndim == 2):    
            assert (a.shape == b.shape), "The input must have the same shape."
        else:    
            assert (a.shape[:1]==b.shape[:1]), "The input must have the same shape."

        tmp = np.fft.ifft2( np.fft.fft2(a, axes=(0,1,))*np.fft.fft2(b[::-1].conj(), axes=(0,1,)), axes=(0,1,))
        if (is_complex(a)+is_complex(b)==0):
            tmp = np.real(tmp)
        return tmp
    else:
        Fa = np.fft.fft2(a, axes=(0,1,))
        tmp = np.fft.ifft2( np.abs(Fa)**2, axes=(0,1,))
        if (~is_complex(a)):
            tmp = np.real(tmp)
        return tmp


def imresample(im, newsize):
    """
    Resample a 2D array to desired size.
    """
    if isinstance(im, np.ndarray):
        if is_complex(im):
            tmp = Image.fromarray(np.real(im))
            im2r = np.array(tmp.resize(newsize))
            tmp = Image.fromarray(np.imag(im))
            im2i = np.array(tmp.resize(newsize))
            return im2r + 1j*im2i
        else:
            tmp = Image.fromarray(im)
    else:
        tmp = im.copy()
    im2 = np.array(tmp.resize(newsize))
    return im2

def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = convolve(image, ar.conj())
    
    image = convolve(np.square(image), a1) - \
            np.square(convolve(image, a1)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

# ################################################################################################
# ###### TV based denoising
# ################################################################################################

# def TV_I (x):
#     a = np.sum( (x[:-1,:]-x[1:,:])**2 + np.transpose((x[:,:-1]-x[:,1:])**2) )
#     b = np.sum( np.abs(x[:-1,-1]-x[1:,-1]) )
#     c = np.sum( np.abs(x[-1,:-1]-x[-1,1:]) )
#     return a+b+c

# def pad_10(array):
#     return np.pad(array,((1,0),(0,0)))
# def pad10(array):
#     return np.pad(array,((0,1),(0,0)))
# def pad0_1(array):
#     return np.pad(array,((0,0),(1,0)))
# def pad01(array):
#     return np.pad(array,((0,0),(0,1)))

# def L(p,q):
#     return pad_10(p)+pad0_1(q)-pad10(p)-pad01(q)

# def L_star(x) :
#     p = x[:-1,:] - x[1:,:]
#     q = x[:,:-1] - x[:,1:]
#     return np.array([p,q],dtype=object)

# def proj_P(pq,m,n, TV_type = 'isotropic') :
#     """ Projection on the set P as defined in the paper.
#         force (p_{i,j}, q_{i,j}) to obey p_{i,j}^2 + q_{i,j}^2 <= 1
    
#         Inputs : (p,q) = (x,y)-like weights for computing the norm 
#         output : (r,s) 
#     """
#     p,q = pq[0], pq[1]
    
#     " Initialization "
#     r = np.zeros((m-1,n), dtype=complex )
#     s = np.zeros((m,n-1), dtype=complex )
    
#     if (TV_type == 'isotropic'):
#         sqrt_sum = np.sqrt(np.abs(p[:,:-1])**2+np.abs(q[:-1,:])**2)
#         sqrt_sum[sqrt_sum<1] = 1
        
#         r[:,:-1] = p[:,:-1]/sqrt_sum
#         den_p_in = np.abs(p[:,-1])
#         den_p_in[den_p_in<1]=1
#         r[:,-1] = p[:,-1]/den_p_in
#         s[:-1,:] = q[:-1,:]/sqrt_sum
#         den_q_in = np.abs(q[-1,:])
#         den_q_in[den_q_in<1]=1
#         s[-1,:] = q[-1,:]/den_q_in
#     else: # "Anisotropic TV"
#         den_p = np.abs(p)
#         den_p[den_p<1] = 1
#         den_q = np.abs(q)
#         den_q[den_q<1] = 1
#         r = p/den_p
#         s = q/den_q
#     return r,s

# def proj_C(x, absxmin=None, absxmax=None) :
#     """ Projects signal x onto a set C, in this case a ring in the complex plane. 
    
#         Inputs : x = signal to be projected
#                 absxmin = min absolute value
#                 absxmax = max absolute value
#         output : x* = projection on C
#     """
#     abs_x = np.abs(x)
#     if (absxmax is not None):
#         ind = abs_x>absxmax
#         x[ind]*=np.sqrt(absxmax/abs_x[ind])
#     if (absxmax is not None):
#         ind = abs_x<absxmin
#         x[ind]*=np.sqrt(absxmin/abs_x[ind])
#     return x


# def GP(b, lamb, N, absxmin=None, absxmax=None):
#     """ Implements the GP algorithm 
    
#         Inputs : b = observe image, lamb = regularization parameter, N = number of iterations
#         output : x* = an optimal solution of min_{x in C} ||x-b||_F^2 + 2.lamb.TV(x)
#     """
    
#     # Initialization
#     m, n = b.shape
#     p_k_1 = np.zeros((m-1,n), dtype=complex)
#     q_k_1 = np.zeros((m,n-1), dtype=complex)
        
#     for k in np.arange(N):
#         p_k, q_k = proj_P( np.array([p_k_1,q_k_1],dtype=object) \
#                           + 1/(8*lamb)* L_star( proj_C(lamb*L(p_k_1, q_k_1), absxmin, absxmax)-b), m, n, TV_type = 'isotropic' )

#         " update "
#         p_k_1, q_k_1 = p_k, q_k
        
#     x_k = b-lamb*L(p_k,q_k)
#     print('f(x_k) = ', np.real(2*lamb*TV_I(x_k)+ np.linalg.norm(x_k-b)**2))
        
#     return proj_C(b-lamb*L(p_k,q_k), absxmin, absxmax)


# def FGP(b, lamb, N, absxmin=None, absxmax=None):
#     """ Implements the FGP algorithm 
    
#         Inputs : b = observe image, lamb = regularization parameter, N = number of iterations
#         output : x* = an optimal solution of min_{x in C} ||x-b||_F^2 + 2.lamb.TV(x)
#     """
    
#     # Initialization
#     m, n = b.shape
#     r_k, p_k_1 = np.zeros((m-1,n), dtype=complex), np.zeros((m-1,n), dtype=complex) 
#     s_k, q_k_1 = np.zeros((m,n-1), dtype=complex), np.zeros((m,n-1), dtype=complex)
#     t_k_1 = 1
        
#     for k in np.arange(N):
#         p_k, q_k = proj_P( np.array([r_k,s_k],dtype=object) \
#                           + 1/(8*lamb)* L_star( proj_C(lamb*L(r_k, s_k), absxmin, absxmax)-b), m, n, TV_type = 'isotropic' )

#         " Update ``timestep´´ " 
#         t_k = (1+ np.sqrt(1+4*t_k_1**2))/2   
        
#         " Gradient descent step "
#         r_k_plus1, s_k_plus1 = np.array([p_k, q_k],dtype=object) \
#                                 + ((t_k-1)/t_k) * np.array([p_k-p_k_1,q_k-q_k_1],dtype=object)

#         " update "
#         t_k_1 = t_k
#         p_k_1, q_k_1 = p_k, q_k
#         r_k, s_k = r_k_plus1, s_k_plus1
        
#     x_k = b-lamb*L(p_k,q_k)
#     print('f(x_k) = ', np.real(2*lamb*TV_I(x_k)+ np.linalg.norm(x_k-b)**2))
         
#     return proj_C(b-lamb*L(p_k,q_k), absxmin, absxmax)

# ################################################################################################
# ###### FISTA
# ################################################################################################

# def FISTA (x0, L, niter = 10, absxmin=None, absxmax=None):
#     y_k = x0
#     x_k_1 = x0
#     t_k_1 = 1
    
#     lamb = np.sqrt(L/16)
    
#     for k in np.arange(niter):
#         x_k = FGP(y_k, lamb, 4, absxmin, absxmax)
#         t_k = (1+np.sqrt(1+4*t_k_1**2))/2
#         y_k = x_k + ((t_k_1-1)/t_k)* (x_k-x_k_1)
        
#         "Update"
#         t_k_1 = t_k
#         x_k_1 = x_k
#     return y_k