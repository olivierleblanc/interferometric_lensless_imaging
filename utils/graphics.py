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
    Date : 03/03/2021

    Code description :
    __________________
    Contains useful functions for graphical processing and rendering. 

"""
import numpy as np
import matplotlib.pyplot as plt

from functions import snr

def set_plot_params (usetex=True, font="Cambria", lgd_sz=12, figsize=(6,6), axlabel_sz=16,axtitle_sz=14, xtick_label_sz=12, ytick_label_sz=12, linewidth = 3, cmap='viridis'):
    """Set parameters for all plots of file.

    Args:
        ...

    Returns:
        /
    """

    plt.rcParams.update({
        "text.usetex": usetex, # LaTeX rendering
        "font.family": font,
        "font.serif": ["Palatino"],
        'legend.fontsize': lgd_sz,
        'figure.figsize': figsize,
        'axes.labelsize': axlabel_sz,
        'axes.titlesize': axtitle_sz,
        'xtick.labelsize': xtick_label_sz,
        'ytick.labelsize': ytick_label_sz,
        'lines.linewidth' : linewidth,
        'image.cmap' : cmap
    })

def labels(ax,title=None,xlabel=None,ylabel=None,zlabel=None):
    """ Writes title and axes labels of ax object.

    Args:
        ax (axis) : the axis object.
        title (str)
        xlabel (str)
        ylabel (str)
        zlabel (str)

    Returns:
        /
    """
    if (title is not None):
        ax.set_title(title)
    if (xlabel is not None):
        ax.set_xlabel(xlabel)
    if (ylabel is not None):
        ax.set_ylabel(ylabel)
    if (zlabel is not None):
        ax.set_zlabel(zlabel)
    return None

def subplot_axs(fig,ny,nx, proj3=None):
    """Create all axes for a subplot.

    Args:
        fig (figure)           : the figure onto creating the axes.
        ny (int)               : number of subplots along y-axis.
        nx (int)               : number of subplots along x-axis.
        proj3 (tuple of tuples): contains the indices of axes that need a projection 3D.

    Returns:
        axs (array of axes) : all the created axes.

    """
    axs=[None]*nx*ny
    wx = ((1-0.05)/(1.1*nx))
    wy = (1/(1.5*ny))

    for j in np.arange(ny):
        for i in np.arange(nx):
            axs[j*nx+i] = fig.add_axes([0.05+i/nx, 0.75-j/ny, wx, wy])

    if (proj3 is not None):
        for tup in proj3:
            j,i = tup
            axs[j*nx+i].remove()
            axs[j*nx+i] = fig.add_axes([0.05+i/nx, 0.75-j/ny, wx, wy], projection='3d')
    if (ny>1 and nx>1):
        axs=np.asarray(axs).reshape(ny,nx) # create 2D axis if necessary

    return axs

def changeax (ax, xmag, ymag):
    """Change magnitude of plot axis.

    Args:
        ax (axis)    : the axis
        xmag (double): magnitude of x-axis
        ymag (double): magnitude of y-axis

    Returns:
        /
    """
    if (xmag!=0):
        ax.set_xticklabels(np.round(ax.get_xticks()/xmag))
    if (ymag!=0):
        ax.set_yticklabels(np.round(ax.get_yticks()/ymag))

def scatter3D(xv,yv,zv,cv,thres,subs=1,alpha=0.3,ms=3):
    """Create 3D figure and scatter plot.

    Args:
        xv,yv,zv (3D arrays): x,y,z coordinates of the data points.
        cv (3D array)       : value of the data point, varied with colormap.
        thres (double)      : threshold value for showing data points.
        subs (int)          : subsampling factor.
        alpha (double)      : transparency factor, 0<alpha<1.
        ms (double)         : markersize.

    Returns:
        /

    """
    xvs=xv[::subs,::subs,::subs]
    yvs=yv[::subs,::subs,::subs]
    zvs=zv[::subs,::subs,::subs]
    abs_u_in_s = cv[::subs,::subs,::subs]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(xvs[abs_u_in_s>thres],zvs[abs_u_in_s>thres],yvs[abs_u_in_s>thres],c=abs_u_in_s[abs_u_in_s>thres], alpha=alpha,s=ms,cmap=cm.jet)
    plt.colorbar(scat)
    ax.invert_yaxis()
    plt.axis('off')
    plt.show()

def sliceplot(xv,yv,zv,cv,ax=None, xind=None,yind=None,zind=None,alpha=0.3,ms=3, cmap='viridis'):
    """Create 3D figure and scatter plot.

    Args:
        xv,yv,zv (3D arrays): x,y,z coordinates of the data points.
        cv (3D array)       : value of the data point, varied with colormap.
        ax (axis)           : axis onto plot.
        xind,yind,zind(ints): index of the planes along x,y,z axes.
        alpha (double)      : transparency factor, 0<alpha<1.
        ms (double)         : markersize.
        cmap (colormap)     : colormap.

    Returns:
        /

    """
    nx,ny,nz=xv.shape

    if (xind is None):
        xind=nx//2
    if (yind is None):
        yind=ny//2
    if (zind is None):
        zind=nz//2

    if (ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    "Back"
    # scat1 = ax.scatter(xv[nx//2,ny//2:,:],zv[nx//2,ny//2:,:],yv[nx//2,ny//2:,:],c=cv[nx//2,ny//2:,:], alpha=alpha,s=ms,cmap=cmap)
    # scat1 = ax.scatter(xv[:,ny//2:,nz//2],zv[:,ny//2:,nz//2],yv[:,ny//2:,nz//2],c=cv[:,ny//2:,nz//2], alpha=alpha,s=ms,cmap=cmap)
    # scat1 = ax.scatter(xv[:,ny//2,nz//2:],zv[:,ny//2,nz//2:],yv[:,ny//2,nz//2:],c=cv[:,ny//2,nz//2:], alpha=alpha,s=ms,cmap=cmap)
    "Lower"
    # scat1 = ax.scatter(xv[:nx//2,:ny//2,nz//2],zv[:nx//2,:ny//2,nz//2],yv[:nx//2,:ny//2,nz//2],c=cv[:nx//2,:ny//2,nz//2], alpha=alpha,s=ms,cmap=cmap)
    # scat1 = ax.scatter(xv[:nx//2,ny//2,:nz//2],zv[:nx//2,ny//2,:nz//2],yv[:nx//2,ny//2,:nz//2],c=cv[:nx//2,ny//2,:nz//2], alpha=alpha,s=ms,cmap=cmap)
    "Front"
    scat1 = ax.scatter(xv[nx//2,:ny//2,:zind],zv[nx//2,:ny//2,:zind],yv[nx//2,:ny//2,:zind],c=cv[nx//2,:ny//2,:zind], alpha=alpha,s=ms,cmap=cmap)
    scat1 = ax.scatter(xv[nx//2:,:ny//2,zind],zv[nx//2:,:ny//2,zind],yv[nx//2:,:ny//2,zind],c=cv[nx//2:,:ny//2,zind], alpha=alpha,s=ms,cmap=cmap)
    scat1 = ax.scatter(xv[nx//2:,ny//2,:zind],zv[nx//2:,ny//2,:zind],yv[nx//2:,ny//2,:zind],c=cv[nx//2:,ny//2,:zind], alpha=alpha,s=ms,cmap=cmap)

    plt.colorbar(scat1,ax=ax)
    ax.invert_yaxis()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    ax.axis('off')


def show_all_depths(obj, xv=None, yv=None, W=3, nw=10, cmap='viridis'):
    """Show 3D object by subplotting each z-plane incrementally.

    Args:
        obj (3D array)      : the object to be plotted.
        xv, yv (3D arrays)  : x,y coordinates in 3D space.
        W (float)           : subplot size.
        nw (int)            : number of images along x-axis.
        cmap (colormap)     : colormap.

    Returns:
        /

    """
    fig = plt.figure(figsize=(W,W))
    nz = obj.shape[2]
    if (xv is not None):
        for i in np.arange(nz):
            ax = fig.add_axes([(i%nw)*W/nw, (nz-1-(i//nw))*W/nw, W/nw, W/nw])
            cp = ax.contourf(xv[:,:,i], yv[:,:,i], obj[:,:,i], cmap=cmap)
            ax.set_axis_off()
    else : 
        for i in np.arange(nz):
            ax = fig.add_axes([(i%nw)*W/nw, (nz-1-(i//nw))*W/nw, W/nw, W/nw])
            cp = ax.contourf(obj[:,:,i], cmap=cmap)
            ax.set_axis_off()
    plt.show()

def show_successive_planes(obj, xv,yv,zv,Lx,Ly,Lz,subsampling=5, ax=None):
    """ Plot successive planes of 'obj' in a 3D figure.

    Args:
        obj (3D array)      : object to be shown.
        xv,yv,zv (3D arrays): x,y,z coordinates of all 3d points.
        Lx,Ly,Lz (floats)   : space size along x,y,z axes.
        subsampling (int)   : number of planes to show.
        ax (axis)           : axis onto plot.

    Returns:
        /
    """
    nz = obj.shape[2]
    if (ax is None):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_axes([0.0, 0.0, 0.9, 0.9], projection='3d')
    for i in np.arange(subsampling+1) :
        indexz = min(int(i*nz/subsampling),nz-1)
        cp1 = ax.scatter((i*Lx/2)+xv[:,:,indexz], \
                        (i*Lz)+zv[:,:,indexz], \
                        (i*Ly/2)+yv[:,:,indexz], c=obj[:,:,indexz], \
                        marker='o', s = 5)
    return ax

#_____________________________________________________________________________________
#                       Plotting the reconstruction results
#_____________________________________________________________________________________

def show_objective(ax, objective, labels, log_scale=True, linewidth=1.2):
    """Subplot the objective vs iteration in axis `ax`.

    Args:
        ax (axis) : axis onto plot.
        objective (float array) : objective function values vs iteration
        labels (list) : list of labels for each dimension in `objective`.
        log_scale (bool) : whether to plot in log scale.
        line_width (float) : line width.

    Returns:
        /

    """

    objective=np.real(np.array(objective))
    nplots = objective.ndim

    if nplots==1:
        objective = objective[..., np.newaxis]
    else:
        ax.plot(np.sum(objective, axis=1), label='Global objective')

    if (labels is not None and len(labels)==nplots):
        for i in range(nplots):
            ax.plot(objective[:, i], label=labels[i])
    else:
        for i in range(nplots):
            ax.plot(objective[:, i], label='Objective'+str(i))

    for line in ax.lines:
        line.set_linewidth(linewidth)
    ax.set_ylim(1e-30)
    ax.grid(True)
    ax.set_title('Convergence')
    ax.legend()
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Objective function value')
    if (log_scale):
        ax.set_yscale('log')
    

def show_rec1D(x, xhat, objective=None, log_scale=True, labels=None, show_Fourier=False, linewidth=1.2):
    """Subplot GT x, reconstruction xhat, and eventually the objective vs iteration and Fourier transforms.
       Also print the SNR of reconstruction.

    Args:
        x (1D array)            : Ground truth  
        xhat (1D array)         : Reconstruction of X 
        objective (float array) : objective function values vs iteration
        show_Fourier (bool)     : if True, plot Fourier transforms

    Returns:
        /

    """
    fig=plt.figure(figsize=(12,4))

    if (objective is None):
        axs=subplot_axs(fig, 1,1)

    else:
        objective=np.real(np.array(objective))
        nplots = objective.ndim
        axs = [fig.add_subplot(1, 2, i+1) for i in range(2)]
        show_objective(axs[1], objective=objective, labels=labels, log_scale=log_scale, linewidth=linewidth)


    axs[0].plot(x, 'o', label='Original')
    axs[0].plot(xhat, 'xr', label='Reconstructed')
    axs[0].grid(True)
    axs[0].set_title('Achieved reconstruction')
    axs[0].legend(numpoints=1)
    axs[0].set_xlabel('Signal dimension number')
    axs[0].set_ylabel('Signal value')
    plt.show()

    if (show_Fourier):
        "Compare reconstructions in Fourier domain"
        fig=plt.figure(figsize=(8,3))
        axs = subplot_axs(fig, 1,2)
        axs[0].plot(np.abs(np.fft.fftshift(np.fft.fft2(x))))
        axs[1].plot(np.abs(np.fft.fftshift(np.fft.fft2(xhat))))
        axs[0].set_title('Ground truth FFT')
        axs[1].set_title('Reconstruction FFT')
        plt.show()

    print('SNR: {:.2f} dB'.format(snr(x, xhat)) )


def show_rec2D(X, Xhat, objective=None, show_error=False, show_Fourier=False, linewidth=1.2, prt_SNR=True):
    """Subplot GT X, reconstruction Xhat, and eventually the objective vs iteration and Fourier transforms.
       Also print the SNR of reconstruction.

    Args:
        X (2D array)            : Ground truth  
        Xhat (2D array)         : Reconstruction of X 
        objective (float array) : objective function values vs iteration
        show_Fourier (bool)     : if True, plot Fourier transforms

    Returns:
        /

    """
    flag_obj = (objective is not None)
    n = 2+flag_obj+show_error
    fig=plt.figure(figsize=(12,4))

    axs = [fig.add_subplot(1, n, i+1) for i in range(n)]

    im0=axs[0].imshow(np.real(X), cmap='viridis')
    im1=axs[1].imshow(np.real(Xhat), cmap='viridis')
    # axs[0].invert_yaxis()
    # axs[1].invert_yaxis()
    fig.colorbar(im0, ax=axs[0])
    fig.colorbar(im1, ax=axs[1])
    axs[0].set_title('Ground truth')
    axs[1].set_title('Noiseless reconstruction')

    if flag_obj:
        objective=np.real(np.array(objective))
        if (objective.ndim==2):
            axs[2].plot(objective[:,0], 'b', label='Data fidelity', linewidth=linewidth)
            axs[2].plot(objective[:,1], 'g', label='Regularization', linewidth=linewidth)
            axs[2].plot(np.sum(objective,axis=1), 'r', label='Global objective', linewidth=linewidth)
        else:
            axs[2].plot(objective, 'b', label='Global objective', linewidth=linewidth)
        axs[2].set_title('objective vs iter')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Objective value')
        axs[2].set_yscale('log')
        axs[2].legend()

    if show_error:
        im3 = axs[2+flag_obj].imshow(np.abs(X-Xhat), cmap='viridis')
        fig.colorbar(im3, ax=axs[2+flag_obj])
        axs[2+flag_obj].set_title('Error')

    plt.show()

    if (show_Fourier):
        "Compare reconstructions in Fourier domain"
        fig=plt.figure(figsize=(8,3))
        axs = subplot_axs(fig, 1,2)
        im0=axs[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(X))))
        im1=axs[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(Xhat))))
        fig.colorbar(im0, ax=axs[0])
        fig.colorbar(im1, ax=axs[1])
        axs[0].set_title('Ground truth FFT')
        axs[1].set_title('Reconstruction FFT')
        plt.show()

    if (prt_SNR):
        print('SNR: {:.2f} dB'.format(snr(X, Xhat)) )