"""
    Author : Olivier Leblanc
    Date : 22/09/2022

    Code description :
    __________________
    Try to get the link between the predicted single-pixel values and the measured ones.
"""
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import loadmat 
import mat73
from pyunlocbox import functions, solvers

import sys, os
def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
sys.path.append(os.path.join(updir(__file__,3),'utils'))
sys.path.append(updir(__file__,2))

from interferometric_lensless_imaging import * # ROP projectors, rmdiag,...
from utils_data import *
from utils_files import get_child
from functions import *
from graphics import *

"_____________________________________________________________________"

t0 = time.time()

folder_date='220624'
full_cores = 1
with_object = 1
load_FOV = 1
use_ROI = 0
load_ds_data = 0
reduce_Sm_dlval = 0
warmstart = 1

# Ms = np.linspace(2800, 1e2, 3).astype(int)
Ms = [5000, 500]
M_ds = Ms[0]
new_sz = 256

DataProc = ILI_DataProcessing(Ms, new_sz, 
        folder_date, 
        folder_base = '/CECI/home/users/l/e/leblanco/Experimental_data/',
        full_cores=full_cores,
        with_object=with_object,
        load_ds_data=load_ds_data,
        load_FOV=load_FOV, 
        reduce_Sm_dlval=reduce_Sm_dlval
)

Sm_dlval, ydl_ds, f_ds = DataProc.get_calib_and_obs()
if (reduce_Sm_dlval):
    FSm_dlval_ds, ind0, ind1 = DataProc.get_FSm_dlval_ds(Sm_dlval)

filepath = '/CECI/home/users/l/e/leblanco/LECI/Interferometric_LE/code/Exp_data_analysis/figs/ind_ROI_{}.npy'.format(new_sz)
if (use_ROI):
  ind_ROI = DataProc.get_ind_ROI(filepath)
else:
   DataProc.get_vignetting()

snrs = np.zeros(len(Ms))
sols = np.zeros((len(Ms),new_sz,new_sz))
x0 = np.zeros((new_sz, new_sz))

t1 = time.time()

for i, M_ds in enumerate(Ms):
    if (i>0):
        Sm_dlval = Sm_dlval[:M_ds,:,:]*np.sqrt(Ms[i]/Ms[i-1])
        ydl_ds = ydl_ds[:M_ds]*np.sqrt(Ms[i]/Ms[i-1])

    "The diagless observation operators." 
    if (reduce_Sm_dlval):
        FSm_dlval_ds = FSm_dlval_ds[:M_ds,:]*np.sqrt(Ms[i]/Ms[i-1])
        Adl_op = lambda x: np.real((FSm_dlval_ds.conj()@T(x)[ind0,ind1])*(1200/new_sz)**2)
        Adl_op2 = lambda Xin: np.einsum('mxy,xy->m', Sm_dlval, Xin)*(1200/new_sz)**2

        def Adlt_op(y):
            adj = np.zeros((new_sz,new_sz), dtype=complex)
            adj[ind0,ind1] = (y@FSm_dlval_ds)*(1200/new_sz)**2
            return np.real(T_star(adj))
    else:
      Adl_op = lambda Xin: np.einsum('mxy,xy->m', Sm_dlval, Xin)*(1200/new_sz)**2
      Adlt_op = lambda y: np.real(np.einsum('m,mxy->xy', y, Sm_dlval))*(1200/new_sz)**2

    nudl = eval_nu(np.random.randn(new_sz,new_sz), Adl_op, Adlt_op, nb_iter=50)*1.01
    print('single CPU: nudl = {}'.format(nudl))
    adjoint_test(Adl_op, Adlt_op, [[new_sz, new_sz],[M_ds]])
    print('Observation operators defined.')

    # Reconstruction from experimental observations 
    maxit = 80
    lamb = 1e9

    solver1 = solvers.generalized_forward_backward(step=0.5/nudl, nu=nudl )
    f1 = functions.norm_tv(maxit=80, dim=2, lambda_=lamb)
    f2 = functions.norm_l2(y=ydl_ds, A=Adl_op, At=Adlt_op, nu=nudl, lambda_=0.5)
    f3 = functions.proj_positive()
    # f3 = functions.proj_box(0,1)

    ret1 = solvers.solve([f1, f2, f3], x0, solver1, rtol=None, maxit=maxit, verbosity='HIGH')
    ret1['sol'] /= np.max(ret1['sol'])
    ret1['objective'] /= np.max(ret1['sol'])

    if (use_ROI):
      snr_val = snr(f_ds[ind_ROI], ret1['sol'][ind_ROI])
    else:
      snr_val = snr(f_ds*DataProc.w, ret1['sol']*DataProc.w)
    print('SNR: {:.2f} dB'.format(snr_val))
    # show_rec2D(f_ds, ret1['sol'], objective=ret1['objective'])

    snrs[i] = snr(f_ds[ind_ROI], ret1['sol'][ind_ROI]) 
    sols[i,:,:] = ret1['sol'] 

    if warmstart:
      x0 = np.copy(ret1['sol'])

plt.figure()
plt.imshow(ret1['sol'])
plt.colorbar()
plt.savefig('reconstruction.png')

np.save('reconstruction.npy', ret1['sol'])
np.save('GT.npy', f_ds)

t2 = time.time()
print('Loading time: {:.2f}s, opti time: {:.2f}s'.format(t1-t0, t2-t1))

# folder = "/CECI/home/users/l/e/leblanco/LECI/Interferometric_LE/code/Exp_data_analysis/reconstruction_data/"
# filename = mat_fname[:-4]+"_ds_lamb1e11_niter"+str(maxit)+".npz"
# filename = filename.split('/')[-1]

# try:
#     print('There was already existing data, concatenate!')
#     tmp = np.load(folder+filename, allow_pickle=True)
#     Ms_old, _, _, sols_old = [tmp['arr_'+str(i)] for i in range(4)]
    
#     Ms_new = np.concatenate((Ms, Ms_old))
#     Ms_new, remove_duplicates_indices = np.unique(Ms_new, return_index=True)

#     sols_new = np.concatenate((sols, sols_old))
#     sols_new = sols_new[remove_duplicates_indices]
#     np.savez(folder+filename, Ms_new, ind_ROI, f_ds, sols_new)
# except:
#     print('Solving as new data')
#     np.savez(folder+filename, Ms, ind_ROI, f_ds, sols)
    
# #set_plot_params()
# plt.figure()
# plt.plot(Ms, snrs, 'ro-', linewidth=2.0)
# plt.xlabel('M')
# plt.ylabel('SNR')
# plt.savefig(folder+filename[:-3]+"png")