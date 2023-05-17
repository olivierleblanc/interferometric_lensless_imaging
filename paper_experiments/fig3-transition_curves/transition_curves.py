"""
    Author : Olivier Leblanc
    Date : 02/05/2023

    Code description :
    __________________
    Generates sparse object and recover it 
    by solving an inverse problem from rank-one projections.

"""

from mpi4py import MPI

import numpy as np
import spgl1
from pyunlocbox import functions, solvers

import sys, os
def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
sys.path.append(os.path.join(updir(__file__,3),'utils'))

from interferometric_lensless_imaging import * # ROP projectors, rmdiag,...
from functions import snr, corr_circ
"________________________________________________"

def get_Qs(Vs):
  path = os.path.join(updir(__file__,2),'Q_vis_bijection.npy')
  Q_vis_bijection = np.load(path, allow_pickle=True) 
  aQs = Q_vis_bijection[:,0].astype('int')
  mean_eff_visibility_cardinality = Q_vis_bijection[:,1]
  Qs = [aQs[np.argmin(np.abs(mean_eff_visibility_cardinality-tvis))] for tvis in Vs]
  return Qs

V=240 # Number of visibilities
Q=get_Qs([V])[0] # Number of cores

ntrial = 10 # xNumber of used processes. Number of trials for each pair of parameters
N = 256 # vector size
Ks = np.arange(5, 12)
Ms = np.arange(2, 138, 8)

SNR_target = 80

if (__name__ == '__main__'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('I\'m rank {}'.format(rank))

    folder = 'simu_results/transition_curves/'
    if (rank==0):
        if not os.path.exists(folder):
            os.makedirs(folder)
    filename = 'snrs{:02d}.npy'.format(rank)

    print(filename)

    snrs = np.zeros((len(Ks), len(Ms), ntrial))

    for i,K in enumerate(Ks):
        for j,M in enumerate(Ms):
            for trialnum in range(ntrial):
                print("(K,M,trial,rank)=({},{},{},{})".format(K, M, trialnum, rank))

                "Create the sample space"
                pos = np.random.randint(0, high=N, size=K)
                "Define the sparse object"
                f = np.zeros(N)
                f[pos[:-1]] = np.random.randn(K-1)
                f[pos[-1]] = -np.sum(f)
                
                " Suggestion LJ "
                # for i in range(M):
                #     beta[i,np.random.permutation(np.arange(0,N))[:Q]] = a_ij[i,:]
                " From analytical developments, beware of the np.conj!!"
                a_ij = np.exp(1j*2*np.pi*np.random.rand(M,Q))
                pos_cores = np.random.permutation(np.arange(N//2))[:Q] # random cores locations
                # pos_cores = np.round(np.arange(Q)*(N-1)/(Q-1)).astype(int) # regularly spaced cores locations
                beta = np.zeros((M,N), dtype=complex)
                beta[:, pos_cores] = a_ij
                beta2 = corr_circ(np.conj(beta)) # autocorrelations of beta

                "Define Om = {p_j - p_k, j,k \in [Q]}"
                Om = np.subtract.outer(pos_cores, pos_cores).astype(int)

                opt_tol = np.linalg.norm(f)*10**(-SNR_target/20)
                # bpalg = lambda A, b: spgl1.spg_bp(A, b, opt_tol=opt_tol, verbosity=0)[0]
                # bpdnalg = lambda A, b, sigma: spgl1.spg_bpdn(A, b, sigma, opt_tol=opt_tol, verbosity=0)[0]
                lasso = lambda A, b, tau: spgl1.spg_lasso(A, b, tau, opt_tol=opt_tol)[0]
                A_corr = Autocorr_model(beta2)
                A_corr_r = Op2Real(A_corr)
                y_corr_r = A_corr_r@f
                # x_corr = bpalg(A_corr_r, y_corr_r)
                # x_corr = bpdnalg(A_corr_r, y_corr_r,1e-10)
                x_corr = lasso(A_corr_r, y_corr_r, tau=np.linalg.norm(f,1))

                snrs[i,j,trialnum] = snr(f, x_corr)

    np.save(folder+filename, snrs)

    "the main rank gathers all the results in 'sucmat.npy'"
    if (rank==0):
      
      files = os.listdir(folder)
      files = [file for file in files if file[-4:]=='.npy']
      files.sort()

      data = np.load(files[0])

      for file in files[1:]:
          print(file)
          data = np.concatenate((data, np.load(file)), axis=2)

      sucmat = np.sum(data>35, axis=2)/data.shape[2]

      np.save(os.path.join(folder,'gathered_data.npy'), sucmat)