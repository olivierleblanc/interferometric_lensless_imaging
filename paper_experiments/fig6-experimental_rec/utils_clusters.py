"""
    Author : Olivier Leblanc
    Date : 15/11/2022

    Code description :
    __________________
    Define class to distribute data between processes in a cluster.
"""

import numpy as np
from matplotlib import pyplot as plt
import types
from mpi4py import MPI

def splitNdiffuse(arr, myrank, comm, sqrt_nproc, datatype=MPI.COMPLEX, tag=1):
    """
    Master rank : split the array arr into nproc subarrays and diffuse them to the other processes.
    Other ranks : receive the subarray from the master rank.
    """

    if (myrank==0):   
        nproc = int(sqrt_nproc**2)
        subsz = int(arr.shape[0]//np.sqrt(nproc))
        for ind in np.arange(1, nproc):
            comm.send({'subshape':(subsz,subsz)}, dest=ind) 
            indH = (ind-1)%sqrt_nproc
            indW = (ind-1)//sqrt_nproc
            sub_arr = np.ascontiguousarray(arr[indH*subsz:(indH+1)*subsz, indW*subsz:(indW+1)*subsz])
            comm.Send([sub_arr,2*sub_arr.size, datatype], dest=ind, tag=tag)
        my_arr = arr[-subsz:, -subsz:]
        return my_arr
    else:
        subshape = comm.recv(source=0)['subshape']
        my_arr = np.empty(subshape)
        comm.Recv([my_arr,2*subshape[0]*subshape[1], datatype], source=0, tag=tag)
        return my_arr

def splitNdiffuse2(tensor, myrank, comm, sqrt_nproc, datatype=MPI.COMPLEX, tag=1):
    """
    Master rank : split the tensor tensor into nproc subtensors and diffuse them to the other processes.
    Other ranks : receive the subtensor from the master rank.
    """
    return None

def recvNsum(my_arr, myrank, comm, nproc, datatype=MPI.COMPLEX, tag=1):
    """
    Slave ranks: send their array to the master rank.
    Master rank: sum the received arrays.
    """
    if (myrank!=0):
        comm.Send([my_arr,2*my_arr.size, datatype], dest=0, tag=tag)
        return np.zeros_like(my_arr)
    else:
        arr = my_arr.copy()
        for ind in np.arange(1, nproc):
            comm.Recv([my_arr,2*my_arr.size, datatype], source=ind, tag=tag)
            arr += my_arr
        return arr

def recvNplace(my_arr, myrank, comm, sqrt_nproc, datatype=MPI.COMPLEX, tag=1):
    """
    Slave ranks: send their array to the master rank.
    Master rank: place the received subarrays in the correct bigger array.
    Assumption: my_arr is square matrix.
    """
    if (myrank!=0):
        comm.Send([my_arr,2*my_arr.size, datatype], dest=0, tag=tag)
        return None
    else:
        nproc = int(sqrt_nproc**2)
        subsz = my_arr.shape[1]
        if my_arr.ndim==2:
            arrshape = (my_arr.shape[0]*sqrt_nproc, my_arr.shape[1]*sqrt_nproc)
        elif my_arr.ndim==3:
            arrshape = (my_arr.shape[0], my_arr.shape[1]*sqrt_nproc, my_arr.shape[2]*sqrt_nproc)
        arr = np.empty(arrshape)
        tmp = np.empty(my_arr.shape)
        for ind in np.arange(1, nproc):
            indH = (ind-1)%sqrt_nproc
            indW = (ind-1)//sqrt_nproc
            comm.Recv([tmp,2*tmp.size, datatype], source=ind, tag=tag)
            if my_arr.ndim==2:
                arr[indH*subsz:(indH+1)*subsz, indW*subsz:(indW+1)*subsz] = tmp
            elif my_arr.ndim==3:
                arr[:, indH*subsz:(indH+1)*subsz, indW*subsz:(indW+1)*subsz] = tmp
        if my_arr.ndim==2:
            arr[-subsz:, -subsz:] = my_arr
        elif my_arr.ndim==3:
            arr[:,-subsz:, -subsz:] = my_arr
        return arr



def eval_nu(init, comm, myrank, sqrt_nproc, my_A, my_A_star=None, nb_iter = 10):
    """Estimate the square norm of the operator B (i.e. ||B||^2) thanks to the power method.
        Useful to bound the norm of B, i.e. give |B(x)|^2 \leq \nu |x|^2`.
    """ 
    nproc = int(sqrt_nproc**2)

    if isinstance(my_A, types.FunctionType):
        if (my_A_star is None):
            raise ValueError("A_star must be provided")
        my_A_op = my_A
        my_At_op = my_A_star
    else:
        if (my_A_star is None):
            my_A_star = my_A.T
        my_A_op = lambda x: my_A@x
        my_At_op = lambda y: my_A_star@y

    if (myrank==0):
        u = init
    for k in range(nb_iter):
        if (myrank==0):
            u = u/np.linalg.norm(u) # Normalize current matrix
        else:
            u=None
        my_u = splitNdiffuse(u, myrank, comm, sqrt_nproc, datatype=MPI.FLOAT, tag=12)
        my_y = my_A_op(my_u)
        y = recvNsum(my_y, myrank, comm, nproc, datatype=MPI.FLOAT, tag=13)
        comm.Bcast([y,2*y.size, MPI.FLOAT], root=0)
        my_u = my_At_op(y)
        u = recvNplace(my_u, myrank, comm, sqrt_nproc, datatype=MPI.FLOAT, tag=1)  
    if (myrank==0):
        return np.linalg.norm(u)
    else:
        return 0


# Compute the reconstruction on the master process
def get_objective(x, smooth_funs, non_smooth_funs):
    obj_smooth = [f.eval(x) for f in smooth_funs]
    obj_nonsmooth = [f.eval(x) for f in non_smooth_funs]
    return obj_nonsmooth + obj_smooth

def GFB_algo(sol, z, smooth_funs, non_smooth_funs, step, lambda_=1):
    """        
        with f_i the differentiable functions and g the non-differentiable, and writing γ=self.step:
        x^{k+1} = prox_{γg} (x^k-γ (∑_i ∇f_i(x^k)) 
    """

    # Smooth functions.
    grad = np.zeros_like(sol)
    for f in smooth_funs:
        grad += f.grad(sol)

    # Non-smooth functions.
    if not non_smooth_funs:
        sol[:] -= step * grad  # Reduces to gradient descent.
    else:
        # z = []
        # for f in non_smooth_funs:
        #     z.append(np.array(x0, copy=True))
        sol2 = np.zeros_like(sol)
        for i, g in enumerate(non_smooth_funs):
            tmp = 2 * sol - z[i] - step * grad
            tmp[:] = np.ascontiguousarray(g.prox(tmp, step * len(non_smooth_funs)))
            z[i] += lambda_ * (tmp-sol)
            sol2 += 1. * z[i] / len(non_smooth_funs)
        sol[:] = sol2
    return sol, z
    

class norm_l2_MPI:
    def __init__(self, A, At, new_sz, ydl_ds, nproc, comm, lambda_=0.5, datatype=MPI.FLOAT):
        # Constructor takes keyword-only parameters to prevent user errors.
        self.A = A # The forward operator of master process
        self.At = At # The adjoint operator of master process
        self.new_sz = new_sz
        self.ydl_ds = ydl_ds # The diagless observations
        self.nproc = nproc # Number of processes
        self.sqrt_nproc = int(np.sqrt(self.nproc)) 
        self.comm = comm # The communicator for the MPI process
        self.lambda_ = lambda_
        self.datatype = datatype
        self.verbosity = None

    def gather_fwd(self, x, bcast=False):
        self.Ax = self.A(x) # Start with the forward operator of master process
        Ai_sub_sol = np.zeros_like(self.ydl_ds)
        for i in np.arange(1, self.nproc):
            self.comm.Recv([Ai_sub_sol,2*Ai_sub_sol.size, MPI.FLOAT], source=i, tag=6)
            self.Ax += Ai_sub_sol # Exploit linearity and add the forward operator of the other processes
        if bcast:
            Ax_y = self.Ax - self.ydl_ds
            for i in np.arange(1, self.nproc):
                self.comm.Send([Ax_y, 2*Ax_y.size, MPI.FLOAT], dest=i, tag=7)

    def gather_adjoint(self, y):
        Aty_master = self.At(y) # Start with the forward operator of master process
        subsz = Aty_master.shape[0]
        Aty = np.zeros((self.new_sz, self.new_sz))
        Aty[-subsz:,-subsz:] = Aty_master
        tmp2 = np.zeros_like(Aty_master)
        for ind in np.arange(1, self.nproc):
            indH = (ind-1)%self.sqrt_nproc
            indW = (ind-1)//self.sqrt_nproc
            self.comm.Recv([tmp2,2*tmp2.size, self.datatype], source=ind, tag=8)
            Aty[indH*subsz:(indH+1)*subsz, indW*subsz:(indW+1)*subsz] = tmp2 # Exploit linearity and add the forward operator of the other processes
        return Aty

    def eval(self, x):
        sol = self.Ax - self.ydl_ds
        # print('eval = {}'.format(self.lambda_ * np.sum(sol**2)))
        return self.lambda_ * np.sum(sol**2)

    def grad(self, x):
        err = self.Ax - self.ydl_ds
        return 2 * self.lambda_ * self.gather_adjoint(err)