"""
    Author : Olivier Leblanc
    Date : 04/11/2022

    Code description :
    __________________
    Define class to load the ILI data.
"""

from mpi4py import MPI

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat 
import mat73
import random

import sys, os
def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
sys.path.append(os.path.join(updir(__file__,3),'utils'))

from utils_files import get_child
from utils_clusters import recvNplace, splitNdiffuse, splitNdiffuse2, get_objective, GFB_algo
from functions import *
from interferometric_lensless_imaging import T


class ILI_DataProcessing:
    def __init__(self, Ms, new_sz=256,
        folder_date='220627',
        folder_base = 'D:/',
        full_cores = 1,
        with_object = 1,
        load_ds_data = 0,
        load_FOV = 1,
        reduce_Sm_dlval = 0,
        comm = None
    ):
        self.Ms = Ms
        self.M_ds = Ms[0]
        self.new_sz = new_sz

        # Options
        self.full_cores = full_cores
        self.with_object = with_object
        self.load_ds_data = load_ds_data
        self.load_FOV = load_FOV
        self.reduce_Sm_dlval = reduce_Sm_dlval
        self.comm = comm

        if (self.comm is not None):
            self.myrank = self.comm.Get_rank()
            self.nproc = self.comm.Get_size()
            self.sqrt_nproc = int(np.sqrt(self.nproc)) 
            self.subsz = int(new_sz//self.sqrt_nproc)

        self.folder_date = folder_date
        self.folder = [x for x in get_child(folder_base) if self.folder_date in x][0]
        mat_fullcores = {
        "220624": 110,
        "220627": 103,
        "220628": 103
        }
        mat_objname = {
        "220624": '_object3',
        "220627": '_objectN',
        "220628": '_objectTarget'
        }
        keywords = [str(mat_fullcores[folder_date])+'cores' if full_cores else '55cores', mat_objname[folder_date] if with_object else '_NOobject']
        self.mat_fname = [x for x in get_child(self.folder) if (all([y in x for y in keywords])and x[-4:]=='.mat') ][0]
        # if (folder_date=='220624'):
        #     self.mat_fname = self.folder+r'/refImg_TMintegrTime.mat'


    def get_calib_and_obs(self):

        if (self.load_ds_data):
            "Directly load subsampled data."
            ds_data_fname = self.mat_fname[:-4]+"_ds.npz"
            tmp = np.load(ds_data_fname, allow_pickle=True)
            compensated_fringes_ds, f_ds, Sm, dlval, y_ds, ind_ROI =\
                [tmp['arr_'+str(i)] for i in range(6)]
            del tmp
            self.new_sz = compensated_fringes_ds.shape[0]

            M = len(y_ds)
            Sm = Sm[:self.M_ds,:,:]*np.sqrt(self.M_ds/M)
            dlval = dlval[self.M_ds,:,:]*np.sqrt(self.M_ds/M)
            y_ds = y_ds[self.M_ds]

            Sm_dlval = Sm-dlval
            y_ds /= np.sqrt(self.M_ds)
            ydl_ds = y_ds-np.mean(y_ds)
        else:
            if (self.comm is None):
                mat, TM = self.load_mat()
                compensated_fringes, y = self.compensate(mat, TM)
                f_ds = self.subsample(compensated_fringes)
                my_Sm_dlval, ydl_ds = self.get_Sm_dlval(mat, y)
            else:
                if (self.myrank == 0):
                    print('I\'m proc 0, I\'m loading the data.')
                    mat, TM = self.load_mat()
                    compensated_fringes, y = self.compensate(mat, TM)
                    f_ds = self.subsample(compensated_fringes)
                else:
                    mat, y, f_ds = None, None, None
                my_Sm_dlval, ydl_ds = self.get_Sm_dlval_MPI(mat, y)

        return my_Sm_dlval, ydl_ds, f_ds
        

    def load_mat(self):
        "Load the mat data" 

        print('Start loading data.')

        "Load recorded data and subsample later."
        TM = mat73.loadmat(os.path.join(self.folder,'TM.mat')) # Transmission matrix

        try:
            mat = mat73.loadmat(self.mat_fname)
        except:
            mat = loadmat(self.mat_fname)
        print('Data loaded.')
        return mat, TM

    def compensate(self, mat, TM):
        """Compensate the fringes, i.e. remove the phase differences and the reference core influence.
        Reminder: one uses a calibration method to try estimating the complex wavefield emitted by each core. However, the normalisation by the reference core is done with a captured reference image. This camera gain used for the reference image is not necessarily the same as for the imaging experiment. Therefore, it is unfeasible to match the amplitudes of the predicted and measured speckle.  
        """

        # (x_{focus}, y_{focus}) is the optical center in the camera plane.
        # y = \{ \alpha_m^* I_\Omega \alpha_m \}_{m=1}^M$ is the vector of all scalar obsevations.
        if (self.folder_date=='220316'): # TO BE VERIFIED
            yfocus = int(mat['yfocus'][0][0])
            xfocus = int(mat['xfocus'][0][0])
        else:
            yfocus = int(mat['yfocus'])
            xfocus = int(mat['xfocus'])

        y = mat['Iomega']
        if (len(y)==1):
            y=y[0]

        """
            Compensates for the phases differences between the cores wrt $(x_focus, y_focus)$
            - S_tensor is the 3D tensor S_{xyq} where S_{::q} contains the complex wavefield emitted by core q.
        """
        S_tensor = TM['TM']
        S_tensor /= np.sqrt(S_tensor.shape[0]) # Compensate for a scaling factor of N^2 due to fft

        refImg_fname = [x for x in get_child(self.folder) if 'refImg_TM' in x][0]
        refImg_mat = loadmat(refImg_fname) # Reference image
        refImg = refImg_mat['refImg']
        E0 = np.sqrt(refImg)
        compensated_fringes = np.einsum('ijk,ij,k->ijk', S_tensor, 1/E0, np.exp(1j*(-np.angle(S_tensor[yfocus,xfocus,:]))))

        # Remove unactivated cores.
        self.highCores_ind = mat['Mask_speckle']['actCoresIndx'].astype(int)-1
        if (self.folder_date=='220627' and self.full_cores): # An exception...
            self.highCores_ind = mat73.loadmat(self.folder+'/coreChoice_103active_17inactive.mat')['highCores'].astype(int)-1

        return compensated_fringes, y

    def subsample(self, compensated_fringes):
        "Subsample the data."

        self.compensated_fringes_ds = np.zeros((self.new_sz, self.new_sz, compensated_fringes.shape[2]), dtype=complex)
        for i in range(compensated_fringes.shape[2]):
            self.compensated_fringes_ds[:,:,i] = imresample(compensated_fringes[:,:,i], (self.new_sz,self.new_sz))
        del compensated_fringes
        self.compensated_fringes_ds = self.compensated_fringes_ds[:,:,self.highCores_ind] 

        if (self.with_object):
            obj_fname = [x for x in get_child(self.folder) if 'through' in x][0]
            object = loadmat(obj_fname)
            f = object['img4']
            f_ds = imresample(f, (self.new_sz, self.new_sz))
        else:
            f_ds = np.ones((self.new_sz, self.new_sz))
        print('Data subsampled.')

        return f_ds

    def get_Sm_dlval(self, mat, y):
        "Compute the Sm and dlval tensors."

        randomPhases = mat['randPhases']
        a_ijm = np.exp(1j*randomPhases.T)
        self.a_ijm = a_ijm[:self.M_ds,self.highCores_ind] # Remove unactivated cores.
        y_ds = y[:self.M_ds]
        y_ds /= np.sqrt(self.M_ds)
        ydl_ds = y_ds-np.mean(y_ds)

        Sm = np.abs( np.einsum('mq,ijq->mij', self.a_ijm, self.compensated_fringes_ds) )**2 /np.sqrt(self.M_ds)/self.new_sz
        dlval = np.einsum('mq,ijq->mij', np.abs(self.a_ijm)**2, np.abs(self.compensated_fringes_ds)**2)/np.sqrt(self.M_ds)/self.new_sz
        Sm_dlval = Sm-dlval

        return Sm_dlval, ydl_ds

    def get_Sm_dlval_MPI(self, mat, y):
        "Compute the Sm and dlval tensors."

        if (self.myrank == 0):
            randomPhases = mat['randPhases']
            a_ijm = np.exp(1j*randomPhases.T)
            self.a_ijm = np.ascontiguousarray(a_ijm[:self.M_ds,self.highCores_ind]) # Remove unactivated cores.
            self.Q = self.a_ijm.shape[1]

            y_ds = np.ascontiguousarray(y[:self.M_ds])
            y_ds /= np.sqrt(self.M_ds)
            ydl_ds = y_ds-np.mean(y_ds)

            my_sub_compensated_fringes = np.ascontiguousarray(self.compensated_fringes_ds[-self.subsz:,-self.subsz:,:])

            _,_,Q = self.compensated_fringes_ds.shape
            for i in range(1, self.nproc):
                indH = (i-1)%self.sqrt_nproc
                indW = (i-1)//self.sqrt_nproc
                sub_compensated_fringes = np.ascontiguousarray( \
                    self.compensated_fringes_ds[indH*self.subsz:(indH+1)*self.subsz, \
                    indW*self.subsz:(indW+1)*self.subsz,:] \
                )
                self.comm.send({'Q':Q}, dest=i, tag=0) 
                # The factor 2 is necessary for complex values.
                self.comm.Send([sub_compensated_fringes,2*sub_compensated_fringes.size,MPI.COMPLEX], dest=i, tag=1) # Here uppercase method is used. More efficient!!

        else: # For the other processes
            data = self.comm.recv(source=0, tag=0)
            Q = data['Q']
            my_sub_compensated_fringes = np.empty((self.subsz, self.subsz, Q), dtype=complex)
            self.comm.Recv([my_sub_compensated_fringes,int(2*self.subsz**2*Q), MPI.COMPLEX], source=0, tag=1)

            # data to be broadcasted later
            y_ds = np.empty(self.M_ds, dtype=float)
            self.a_ijm = np.empty((self.M_ds,Q), dtype=complex)

        ###########################################################################
        # Now, all processes are working on their own subpart of the data.
###########################################################################
        self.comm.Bcast([y_ds,2*len(y_ds),MPI.FLOAT], root=0)
        self.comm.Bcast([self.a_ijm,2*self.a_ijm.size, MPI.COMPLEX], root=0) 
        ydl_ds = y_ds-np.mean(y_ds)

        my_Sm = np.abs( np.einsum('mq,ijq->mij', self.a_ijm, my_sub_compensated_fringes) )**2/np.sqrt(self.M_ds)/self.new_sz
        my_dlval = np.einsum('mq,ijq->mij', np.abs(self.a_ijm)**2, np.abs(my_sub_compensated_fringes)**2)/np.sqrt(self.M_ds)/self.new_sz
        my_Sm_dlval = my_Sm-my_dlval
        
        return my_Sm_dlval, ydl_ds

    def get_FSm_dlval_ds(self, Sm_dlval):
        "Compute the FT of the Sm_dlval tensor and exploit its sparsity in the Fourier domain."
        FSm_dlval = T(Sm_dlval, axes=(1,2)) # Fourier transform of Sm_dlval
        w = np.linalg.norm(FSm_dlval, axis=0) # Norm of the Fourier transform of my_Sm_dlval
        s = (w/np.linalg.norm(w,'fro'))**2
        svec = s.reshape(-1)
        r = -np.sort(-svec)
        cums = np.cumsum(r)
        nkept = np.where(cums>0.99)[0][0]
        ind = np.unravel_index(np.argsort(-s, axis=None), s.shape)
        ind0 = ind[0][:nkept]
        ind1 = ind[1][:nkept]
        # plt.figure()
        # plt.scatter((ind0+new_sz//2)%new_sz, (ind1+new_sz//2)%new_sz, c='r', s=1)
        FSm_dlval_ds = FSm_dlval[:,ind0,ind1]
        return FSm_dlval_ds, ind0, ind1

    def get_FSm_dlval_ds_MPI(self, my_Sm_dlval):
        "Compute the FT of the Sm_dlval tensor and exploit its sparsity in the Fourier domain."
        FSm_dlval = recvNplace(my_Sm_dlval, self.myrank, self.comm, self.sqrt_nproc, datatype=MPI.FLOAT)

        if (self.myrank==0):
            w = np.linalg.norm(FSm_dlval, axis=0) # Norm of the Fourier transform of my_Sm_dlval
            s = (w/np.linalg.norm(w,'fro'))**2
            svec = s.reshape(-1)
            r = -np.sort(-svec)
            cums = np.cumsum(r)
            nkept = np.where(cums>0.99)[0][0]
            print(nkept)
            ind = np.unravel_index(np.argsort(-s, axis=None), s.shape)
            ind0 = np.ascontiguousarray(ind[0][:nkept])
            ind1 = np.ascontiguousarray(ind[1][:nkept])
            FSm_dlval_ds = np.ascontiguousarray(FSm_dlval[:,ind0,ind1])
        else: 
            ind0 = None
            ind1 = None
            FSm_dlval_ds = None
        
        my_FSm_dlval_ds = splitNdiffuse2(FSm_dlval_ds, self.myrank, self.comm, self.sqrt_nproc, datatype=MPI.FLOAT)

        return my_FSm_dlval_ds, ind0, ind1

    def split_ind(self, ind0, ind1):
        "Distribute the indices composing FSm_dlval between the different processes."

        if self.myrank==0:
            nkept = len(ind0)
            for i in range(1, self.nproc):
                self.comm.send({'nkept':nkept}, dest=i, tag=0) 
        else: # For the other processes
            data = self.comm.recv(source=0, tag=0)
            nkept = data['nkept']
            my_ind0 = np.empty(nkept, dtype=int)
            my_ind1 = np.empty(nkept, dtype=int)
        if self.myrank==0:
            for i in range(1, self.nproc):
                self.comm.Send([ind0, 2*nkept, MPI.INT], dest=i, tag=1)
                self.comm.Send([ind1, 2*nkept, MPI.INT], dest=i, tag=2)
            my_ind0 = np.copy(ind0)
            my_ind1 = np.copy(ind1)
        else:
            self.comm.Recv([my_ind0, 2*nkept, MPI.INT], source=0, tag=1)
            self.comm.Recv([my_ind1, 2*nkept, MPI.INT], source=0, tag=2)
        indH = (self.myrank-1)%self.sqrt_nproc
        indW = (self.myrank-1)//self.sqrt_nproc
        my_ind0 -= indH*self.subsz
        my_ind1 -= indW*self.subsz
        # sel = np.where((my_ind0>=0) & (my_ind0<self.subsz) & (my_ind1>=0) & (my_ind1<self.subsz))[0]
        # my_ind0 = my_ind0[sel]
        # my_ind1 = my_ind1[sel]
        # my_ind0 = my_ind0[sel]
        # my_ind1 = my_ind1[sel]
        my_ind0 = my_ind0[my_ind0>=0]
        my_ind1 = my_ind1[my_ind1>=0]
        my_ind0 = my_ind0[my_ind0<self.subsz]
        my_ind1 = my_ind1[my_ind1<self.subsz]

        return my_ind0, my_ind1


    def apply_GFB(self, x0, my_Adl_op, my_Adlt_op, nudl, smooth_funs, non_smooth_funs, maxit=30, verbosity='LOW'):
        rtol = None 
        atol = None 
        xtol = None 
        dtol = None 
        rtol_only_zeros = True

        sol = np.copy(x0)
        M_ds = len(my_Adl_op(sol[:self.subsz, :self.subsz]))
        funs = smooth_funs + non_smooth_funs

        if self.myrank==0:
            # Set functions verbosity 
            functions_verbosity = []
            for f in funs:
                functions_verbosity.append(f.verbosity)
                f.verbosity = verbosity

            if verbosity == 'HIGH':
                print('INFO: Generalized forward-backward minimizing {} smooth '
                        'functions and {} non-smooth functions.'.format(
                            len(smooth_funs), len(non_smooth_funs)))

        if (self.myrank!=0):
            indH = (self.myrank-1)%self.sqrt_nproc
            indW = (self.myrank-1)//self.sqrt_nproc
            lby = indH*self.subsz
            uby = (indH+1)*self.subsz
            lbx = indW*self.subsz
            ubx = (indW+1)*self.subsz
            y_tmp = np.ascontiguousarray(my_Adl_op(x0[lby:uby,lbx:ubx]))
            self.comm.Send([y_tmp,2*M_ds,MPI.FLOAT], dest=0, tag=6)
        else:
            step=0.5/nudl

            # Evaluate the objective function at the begining            
            smooth_funs[0].gather_fwd(x0[-self.subsz:,-self.subsz:], bcast=False)
            objective = [get_objective(x0, smooth_funs, non_smooth_funs)]
            print('Objective function  = {}'.format(objective))

        z = []
        for f in non_smooth_funs:
            z.append(np.array(x0, copy=True))

        #######################################################################
        # Iterations of the optimization algorithm.
        #######################################################################
        crit = None
        niter = 0
        datatype = MPI.FLOAT
        while not crit:
            data = self.comm.bcast({'niter':niter},root=0)
            if self.myrank!=0: # Instantiate data that master will broadcast
                niter = data['niter']
                sub_sol = np.empty((self.subsz,self.subsz)) 

            if (self.myrank==0):
                print('I\'m proc {0}, iteration {1}'.format(self.myrank, niter))
                niter += 1
            sub_sol = splitNdiffuse(sol, self.myrank, self.comm, self.sqrt_nproc, datatype=MPI.FLOAT, tag=1)           

            if (self.myrank!=0): # Slaves compute their part of the forward operator
                Ai_sub_sol = my_Adl_op(sub_sol)
                self.comm.Send([Ai_sub_sol,2*Ai_sub_sol.size, datatype], dest=0, tag=6)
            else: # Master gathers the forward operators from the slaves
                if xtol is not None:
                    last_sol = np.array(sol, copy=True)

                # Slaves send fwd op to master and master sends back the concatenation    
                smooth_funs[0].gather_fwd(sub_sol, bcast=True)

            if (self.myrank!=0): # Slaves compute their part of the adjoint operator
                Ax_y = np.empty(M_ds)
                self.comm.Recv([Ax_y, 2*Ax_y.size, datatype], source=0, tag=7)
                At_Ax_y = np.ascontiguousarray(my_Adlt_op(Ax_y))
                self.comm.Send([At_Ax_y,2*At_Ax_y.size, datatype], dest=0, tag=8)
            else: # Master gathers the adjoint operators from the slaves
                sol, z = GFB_algo(sol, z, smooth_funs, non_smooth_funs, step)

                objective.append(get_objective(sol, smooth_funs, non_smooth_funs))
                current = np.sum(objective[-1])
                last = np.sum(objective[-2])

                # Verify stopping criteria.
                if atol is not None and current < atol:
                    crit = 'ATOL'
                if dtol is not None and np.abs(current - last) < dtol:
                    crit = 'DTOL'
                if rtol is not None:
                    div = current  # Prevent division by 0.
                    if div == 0:
                        if verbosity in ['LOW', 'HIGH', 'ALL']:
                            print('WARNING: (rtol) objective function is equal to 0 !')
                        if last != 0:
                            div = last
                        else:
                            div = 1.0  # Result will be zero anyway.
                    else:
                        rtol_only_zeros = False
                    relative = np.abs((current - last) / div)
                    if relative < rtol and not rtol_only_zeros:
                        crit = 'RTOL'
                if xtol is not None:
                    err = np.linalg.norm(sol - last_sol)
                    err /= np.sqrt(last_sol.size)
                    if err < xtol:
                        crit = 'XTOL'
                if maxit is not None and niter >= maxit:
                    crit = 'MAXIT'

                if verbosity in ['HIGH', 'ALL']:
                    print('    objective = {:.2e}'.format(current))

            data = self.comm.bcast({'crit':crit},root=0)
            if (self.myrank!=0):
                crit = data['crit']

        if (self.myrank==0):
            # Restore verbosity for functions. In case they are called outside solve().
            for k, f in enumerate(funs):
                f.verbosity = functions_verbosity[k]

            if verbosity in ['LOW', 'HIGH', 'ALL']:
                print('Solution found after {} iterations:'.format(niter))
                print('    objective function f(sol) = {:e}'.format(current))
                print('    stopping criterion: {}'.format(crit))

            return sol
        return None
        

    def get_speckleImg_and_phases(self):
        try:
            mat = mat73.loadmat(self.mat_fname)
        except:
            mat = loadmat(self.mat_fname)
        speckleImg = mat['randSpeckles']
        # speckleImg = imresample(speckleImg, (self.new_sz,self.new_sz))
        randomPhases = mat['randPhases']
        return speckleImg, randomPhases

    def compute_intM(self, f_ds):
        "Compute the interferometric matrix."
        self.intM = np.einsum('ijq,ijp->qp', 
                np.einsum('ijq,ij->ijq', self.compensated_fringes_ds, f_ds), self.compensated_fringes_ds)

    def get_tensor(self, f_ds):
        "Compute the 3D tensor T: T_{:,:,m} = {aa* \circdot intM}_{m=1}^M."
        self.compute_intM(f_ds)
        aaT = np.einsum('mp,mq->pqm', self.a_ijm, self.a_ijm.conj())
        tensor = np.einsum('abc,ab->abc', aaT, self.intM)
        return tensor

    def get_cores_pos(self):
        "Compute the position of the cores in the tensor."
        tmp = np.fft.fftshift(np.abs(np.sum( np.fft.fft2(self.compensated_fringes_ds, axes=(0,1)), axis=2)))
        return tmp

    def get_Fourier_coverage(self):
        "Compute the position of the visibilities being the difference set of the cores positions."
        tmp = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.abs(np.sum(self.compensated_fringes_ds, axis=2))**2, axes=(0,1))))
        return tmp
    
    def get_vignetting(self):
        "Compute the vignetting function by summing the tensor along the third dimension."
        self.w = np.sum( np.abs(self.compensated_fringes_ds)**2, axis=2)
        self.w /= np.max(self.w)

    def get_ind_ROI(self, filepath, visu=False, save=False):
        if self.load_FOV:
            ind_ROI = np.load(filepath)
        else:
            #Estimate the FWHM of the enveloppe to compute the SNR in the ROI.
            mFOV = np.mean(np.abs(self.compensated_fringes_ds)**2,2) # Mean FOV
            mFOV /= np.max(mFOV)

            x = np.arange(self.new_sz)
            xx, yy = np.meshgrid(x,x)
            mu0 = np.sum(xx*mFOV)/np.sum(mFOV)
            mu1 = np.sum(yy*mFOV)/np.sum(mFOV)
            mu = mu0, mu1

            sig = 2e1
            error = np.linalg.norm(mFOV-gaussian2(xx,yy,mu,sig))**2
            h = 1e-4 #step size 
            niter=5000
            errors = np.zeros(niter)

            for i in range(niter):
                val = 2* np.sum((gaussian2(xx,yy,mu,sig)-mFOV)*(gaussian_dsig2(xx,yy,mu,sig)))
                sig -= h*val

                errors[i] = np.linalg.norm(mFOV-gaussian2(xx,yy,mu,sig))**2
                if (errors[i]<1e-5):
                    break

            fit = np.exp( -((xx-mu[0])**2+(yy-mu[1])**2)/(2*sig**2))
            FWHM = 2*np.sqrt(2*np.log(2))*sig
            ind_ROI = np.logical_and(np.logical_and(mu[0]-FWHM/2 < xx, xx < mu[0]+FWHM/2) \
                , np.logical_and(mu[1]-FWHM/2 < yy, yy < mu[1]+FWHM/2))

            if visu:
                fig = plt.figure(figsize=(12,3))
                axs = [fig.add_axes([i*0.25,0.05,0.18, 0.95]) for i in range(4)]
                axs[0].scatter(mu0, mu1, c='r', s=3.0, label='center')
                axs[0].legend()
                im0=axs[0].imshow(mFOV)
                im1=axs[1].imshow(fit)
                im2=axs[2].imshow(np.abs(mFOV-fit)**2)
                fig.colorbar(im0,ax=axs[0])
                fig.colorbar(im1,ax=axs[1])
                fig.colorbar(im2,ax=axs[2])
                "Draw rectangle"
                axs[1].plot([mu[0]-FWHM/2,mu[0]-FWHM/2],[mu[1]-FWHM/2,mu[1]+FWHM/2],'r', label='ROI')
                axs[1].plot([mu[0]-FWHM/2,mu[0]+FWHM/2],[mu[1]-FWHM/2,mu[1]-FWHM/2],'r')
                axs[1].plot([mu[0]+FWHM/2,mu[0]+FWHM/2],[mu[1]-FWHM/2,mu[1]+FWHM/2],'r')
                axs[1].plot([mu[0]-FWHM/2,mu[0]+FWHM/2],[mu[1]+FWHM/2,mu[1]+FWHM/2],'r')
                axs[1].legend()
                axs[3].plot(errors)
                axs[3].set_xlabel('iter')
                axs[3].set_ylabel('error')
                axs[0].set_title('Mean experimental FOV')
                axs[1].set_title('Best gaussian fit')
                axs[2].set_title('Error')
                plt.show()

            if save:
                np.save(filepath, ind_ROI)

        print('ROI has been defined.')
        return ind_ROI

