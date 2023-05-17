"""
    Author : Olivier Leblanc
    Date : 18/01/2023

    Code description :
    __________________
    Generates sparse object and recover it 
    by solving an inverse problem from rank-one projections.

    Kmax = 8, Mmax = Kmax*log(eN/Kmax)=35.7->50, Qmax=60
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
sys.path.append(updir(__file__,2))

from interferometric_lensless_imaging import * # ROP projectors, rmdiag,...
from functions import snr, corr_circ
from list_utils import * # To save the simulation results in txt files.
"________________________________________________"

def get_Qs(Vs):
  path = os.path.join(updir(__file__,2),'Q_vis_bijection.npy')
  Q_vis_bijection = np.load(path, allow_pickle=True) 
  aQs = Q_vis_bijection[:,0].astype('int')
  mean_eff_visibility_cardinality = Q_vis_bijection[:,1]
  Qs = [aQs[np.argmin(np.abs(mean_eff_visibility_cardinality-tvis))] for tvis in Vs]
  return Qs

Keq=4 # sparsity
Veq=240 # Number of visibilities
Qeq=get_Qs([Veq])[0] # Number of cores
Meq=130 # Number of observations y

ntrial = 5 # xNumber of used processes. Number of trials for each pair of parameters
N = 256 # vector size
Ks = np.arange(4, 41, 3)
Ms = np.arange(2, 138, 8)
Vs = np.arange(30,270,30) 
Qs = get_Qs(Vs)

SNR_target = 80

possibilities = ['K', 'V', 'M']
value_sets = [Ks, Vs, Ms]

big_folder = 'simu_results/'

if (__name__ == '__main__'):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  print('I\'m rank {}'.format(rank))

  for indvar in range(3):
    if indvar == 0:
      fixed_str = 'K'
      str0 = 'V'
      str1 = 'M'
      set0 = np.copy(Vs)
      set1 = np.copy(Ms)
      fixed_var = np.copy(Keq)
    elif indvar == 1:
      fixed_str = 'V'
      str0 = 'K'
      str1 = 'M'
      set0 = np.copy(Ks)
      set1 = np.copy(Ms)
      fixed_var = np.copy(Veq)
    elif indvar == 2:
      fixed_str = 'M'
      str0 = 'K'
      str1 = 'V'
      set0 = np.copy(Ks)
      set1 = np.copy(Vs)
      fixed_var = np.copy(Meq)

    folder = big_folder+'{}/{}/'.format(str0+str1,rank)
    prefix = folder+r'{}_fixed/{}/'.format(fixed_str, str(fixed_var))
      
    thelist = get_pairs(set0, set1)
    var0,var1, trialnum = choose_pair_tot(prefix, fixed_str, possibilities, thelist, ntrial)


    while (var0 is not None):
      print("({},{},trial,rank)=({},{},{},{})".format(str0, str1, var0, var1, trialnum, rank))

      if indvar==0:
        K = np.copy(fixed_var)
        V = np.copy(var0)
        M = np.copy(var1)
      elif indvar==1:
        K = np.copy(var0)
        V = np.copy(fixed_var)
        M = np.copy(var1)
      elif indvar==2:
        K = np.copy(var0)
        V = np.copy(var1)
        M = np.copy(fixed_var)
      Q = get_Qs([V])[0]
    
      "Create the sample space"
      pos = np.random.randint(0, high=N, size=K)
      "Define the sparse object"
      f = np.zeros(N)
      indices = np.random.permutation(np.arange(N))[:K] 
      f[indices[:-1]] = np.random.randn(K-1)
      f[indices[-1]] = -np.sum(f)
    
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
      "Compute the multiplicity of each frequency"
      multiplicities = ind_multiplicity_1D(Om)

      Om_up = Om[np.triu_indices(Om.shape[0],k=1)]
      unique_Om = np.unique(Om_up)
      eff_visibility_cardinality = unique_Om.shape[0]
      # print('There are {} unique frequencies compared to the maximum possible value Q(Q-1)/2={}'.format(unique_Om.shape[0], int(Q*(Q-1)/2)))

      "Compute the interferometric matrix"
      IntM = S_Om_1D(T(f), Om, mult_mat=multiplicities)
    
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

      "Compute the interferometric matrix"
      IntM_hat = S_Om_1D(T(x_corr), Om, mult_mat=multiplicities)
      print('SNR: {:.2f} dB,       SNR on IntM: {:.2f} dB'.format(snr(f, x_corr), snr(IntM, IntM_hat)) )
    
    
      norms = [np.linalg.norm(f), np.linalg.norm(f-x_corr), np.linalg.norm(IntM), np.linalg.norm(IntM-IntM_hat)]
      variables = np.array([K, V, M])
      save_data(fixed_str, possibilities, variables, folder, norms, eff_visibility_cardinality)
    
      var0,var1, trialnum = choose_pair_tot(prefix, fixed_str, possibilities, thelist, ntrial)

  """
  Once all the ranks finished their job, all the data is gathered in a single directory.
  """

  if (rank != 0):
    comm.send(rank, dest=0, tag=rank)
  else:
    for i in range(1, comm.Get_size()):
      comm.recv(source=i, tag=i) # Supposed to be blocking
    print('Start Gathering Data!')

    # import ast
    import shutil

    from list_utils import append_dirs, is_in_dir, is_in_pairs

    K=Ks[0]
    V=Vs[-1]
    M=Ms[-1]
    variables = np.array([K, V, M])

    for indvar in range(3):
      if indvar == 0:
        str0 = 'V'
        str1 = 'M'
      elif indvar == 1:
        str0 = 'K'
        str1 = 'M'
      elif indvar == 2:
        str0 = 'K'
        str1 = 'V'

      curfolder = big_folder+r'{}/'.format(str0+str1)

      fixed_str = possibilities[indvar]
      fixed_var = variables[indvar]
      prefix_appened = curfolder+r'0/{}_fixed/{}/'.format(fixed_str, str(fixed_var))
      dest = curfolder + r'{}_fixed/{}/'.format(fixed_str, str(fixed_var))

      if (os.path.exists(dest) is False):
              os.makedirs(dest, exist_ok=True)

      count=0
      for subdir in os.listdir(curfolder):
        if (subdir.isnumeric()):
          prefix_source = curfolder+r'{}/{}_fixed/{}/'.format(subdir, fixed_str, str(fixed_var))
            
          print("Taking data from rank {}".format(subdir))
        
          if (count==0):
            append_dirs(prefix_source, prefix_appened, dest, fixed_str, possibilities)
            count=-1 # Not needed anymore
          else:
            append_dirs(prefix_source, dest, dest, fixed_str, possibilities)
            
          shutil.rmtree(curfolder+r'{}/'.format(subdir))
