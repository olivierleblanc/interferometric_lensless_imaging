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
    Date : 21/09/2021

    Code description :
    __________________
    Generates sparse object and recover it 
    by solving an inverse problem from rank-one projections.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import random

import sys, os
def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
sys.path.append(updir(__file__,2))

from list_utils import * # To save the simulation results in txt files.
from get_transition_matrix import get_transition_matrix

plt.rcParams.update({
    "text.usetex": True, # LaTeX rendering
    "font.family": 'CMU Serif'
})
fs = 20 # font size for axis labels
fs_tick = 11 # font size for ticks


class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines()+self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        zaxis = self.zaxis

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes
"______________________________________________________________"

N=256
K=4
V=240
M=130
variables = [K, V, M]
possibilities = ['K', 'V', 'M']

common_folder = r'simu_results\\'
folder_KM = common_folder+r'KM\\'
folder_KV = common_folder+r'KV\\'
folder_VM = common_folder+r'VM\\'

def get_paths(prefix, fixed_val, possibilities=['K', 'V', 'M']):
    vary_val_str = "".join([val for val in possibilities if val!=fixed_val])
    path_inputs = prefix+vary_val_str+'_values.txt'
    path_GTnorms = prefix+'GTnorms.txt'
    path_errnorms = prefix+'errnorms.txt'
    path_intM_GTnorms = prefix+'intM_GTnorms.txt'
    path_intM_errnorms = prefix+'intM_errnorms.txt'
    paths = [path_inputs, path_GTnorms, path_errnorms, path_intM_GTnorms, path_intM_errnorms]
    return paths


fig = plt.figure(figsize=(16,3.5))
axs = [fig.add_subplot(141, projection='3d')]
axs[0].set_position([0.05, 0.2, 0.18, 0.75])
fig.add_axes(MyAxes3D(axs[0], 'l'))

bottom = 0.2
width = 0.16
height = 0.75
axs += [fig.add_axes([0.32, bottom, width, height]), fig.add_axes([0.54, bottom, width, height]), fig.add_axes([0.76, bottom, width, height]), fig.add_axes([0.94, bottom, 0.01, height])]

###############
##### KM ######
###############
fixed_val='V'

prefix = folder_KM+r'{}_fixed\{}\\'.format(fixed_val, variables[np.where([elem==fixed_val for elem in possibilities])[0][0]])
paths = get_paths(prefix, fixed_val, possibilities)
transmat_KM, intM_transmat_KM = get_transition_matrix(paths)
transmat_KM = transmat_KM.T[:-1,:]

inputs = np.array(read_list_from_txt(paths[0]))

Ks = inputs[:,:,0]
Ms = inputs[:,:,1]
Ms = Ms[0,:]
Ks = Ks[:,0]

"On reconstruction of the object"
axs[1].set_xlabel(r'$K$', fontsize=fs)
axs[1].set_ylabel(r'$M$', fontsize=fs)
im0 = axs[1].imshow(transmat_KM, cmap='gray', origin='lower', aspect='equal')
plt.locator_params(axis='x', nbins=len(Ks))
axs[1].set_xticks(np.arange(len(Ks)))
plt.locator_params(axis='y', nbins=len(Ms))
axs[1].set_yticks(np.arange(len(Ms)))
axs[1].axis('equal')

xlab = Ks.astype(str)
ylab = Ms.astype(str)
for i in range(len(ylab)):
    if (i%2 == 0):
        ylab[i] = ''
axs[1].set_xticklabels(xlab, fontsize=fs_tick)
axs[1].set_yticklabels(ylab, fontsize=fs_tick)
axs[1].set_aspect(0.79)

Ks2 = np.linspace(0, 8.2, 80)
C = 14.0e-1
D = 5
# axs[1].plot(Ks2, C*Ks2*np.log(12*np.exp(1)*N/Ks2), 'r--', linewidth=2.5)
axs[1].plot(Ks2, C*Ks2+D, 'r--', linewidth=2.5)

###############
##### KV ######
###############
fixed_val='M'

prefix = folder_KV+r'{}_fixed\{}\\'.format(fixed_val, variables[np.where([elem==fixed_val for elem in possibilities])[0][0]])
paths = get_paths(prefix, fixed_val)
transmat_KV, intM_transmat_KV = get_transition_matrix(paths)
transmat_KV = transmat_KV[:,:-1]
inputs = np.array(read_list_from_txt(paths[0]))

Vs = inputs[:,:,1]
Ks = inputs[:,:,0]
Ks = Ks[:,0]
Vs = Vs[0,:]

axs[2].set_xlabel(r'$K$', fontsize=fs)
axs[2].set_ylabel(r'$|\mathcal{V}|$', fontsize=fs)
im1 = axs[2].imshow(transmat_KV.T, cmap='gray', origin='lower')
plt.locator_params(axis='x', nbins=len(Ks))
axs[2].set_xticks(np.arange(len(Ks)))
plt.locator_params(axis='y', nbins=len(Vs))
axs[2].set_yticks(np.arange(len(Vs)))
axs[2].axis('equal')

xlab = Ks.astype(str)
ylab = Vs.astype(str)
# for i in range(2,len(xlab)):
#     if (i%2 != 0):
#         xlab[i] = ''
# for i in range(len(ylab)):
#     if (i%2 == 0):
#         ylab[i] = ''
axs[2].set_xticklabels(xlab, fontsize=fs_tick)
axs[2].set_yticklabels(ylab, fontsize=fs_tick)
axs[2].set_aspect(1.67)

Ks3 = np.linspace(0, 12.0, 100)
C3 = 0.3e-3
axs[2].plot(Ks3, C3*Ks3*np.log(N)**4, 'r--', linewidth=2.5)

###############
##### VM ######
###############
fixed_val='K'

prefix = folder_VM+r'{}_fixed\{}\\'.format(fixed_val, variables[np.where([elem==fixed_val for elem in possibilities])[0][0]])
paths = get_paths(prefix, fixed_val)
transmat_VM, intM_transmat_QM = get_transition_matrix(paths)
transmat_VM = transmat_VM[:,:-1]
inputs = np.array(read_list_from_txt(paths[0]))

Ms = inputs[:,:,1]
Vs = inputs[:,:,0]
Vs = Vs[:,0]
Ms = Ms[0,:]

axs[3].set_xlabel(r'$M$', fontsize=fs)
axs[3].set_ylabel(r'$|\mathcal{V}|$', fontsize=fs)
im2 = axs[3].imshow(transmat_VM, cmap='gray', origin='lower')
plt.locator_params(axis='x', nbins=len(Ms))
axs[3].set_xticks(np.arange(len(Ms)))
axs[3].set_xticklabels(Ms)
plt.locator_params(axis='y', nbins=len(Vs))
axs[3].set_yticks(np.arange(len(Vs)))
axs[3].set_yticklabels(Vs)
axs[3].axis('equal')

xlab = Ms.astype(str)
ylab = Vs.astype(str)
for i in range(len(xlab)):
    if (i%2 == 0):
        xlab[i] = ''
# for i in range(2,len(ylab)):
#     if (i%2 == 0):
#         ylab[i] = ''
axs[3].set_xticklabels(xlab, fontsize=fs_tick)
axs[3].set_yticklabels(ylab, fontsize=fs_tick)
axs[3].set_aspect(2.17)

Vs2 = np.linspace(0.5, 7.5, 100)
axs[3].plot(5.0*np.ones(len(Vs2)), Vs2, 'r--', linewidth=2.5)
Ms2 = np.linspace(5.0, 16.2, 100)
axs[3].plot(Ms2, 0.5*np.ones(len(Ms2)), 'r--', linewidth=2.5)

###############
#### CUBE #####
###############
Ks = np.arange(4, 41, 3)
Vs = np.arange(30, 270, 30)
Ms = np.arange(2,138,8)
Ms = Ms[:-1]

cmap = plt.cm.gray
xplot = axs[0].plot_surface(Ms[-1], Ks[:, np.newaxis], Vs[np.newaxis, :], facecolors=cmap(transmat_KV), shade=False, vmin=0,vmax=1, zorder=1)
yplot = axs[0].plot_surface(Ms[np.newaxis, :], Ks[0], Vs[:, np.newaxis], facecolors=cmap(transmat_VM), shade=False, vmin=0, vmax=1, zorder=2)
zplot = axs[0].plot_surface(Ms[:, np.newaxis], Ks[np.newaxis, :], np.atleast_2d(Vs[-1]), facecolors=cmap(transmat_KM), shade=False, vmin=0,vmax=1, zorder=3)

axs[0].set_xlabel(r'$M$', fontsize=fs, labelpad=8)
axs[0].set_ylabel(r'$K$', fontsize=fs, labelpad=8)
axs[0].zaxis.set_rotate_label(False) # disable automatic rotation
axs[0].set_zlabel(r'$|\mathcal V|$', fontsize=fs, labelpad=10, rotation='horizontal')

axs[0].set_xticks(Ms)
axs[0].set_yticks(Ks)
axs[0].set_zticks(Vs)

xlab = Ms.astype(str)
ylab = Ks.astype(str)
zlab = Vs.astype(str)
for i in range(2,len(xlab)):
    if (i%3 != 0):
        xlab[i] = ''
xlab[1]=''
for i in range(2,len(ylab)):
    if (i%2 == 0):
        ylab[i] = ''
for i in range(2,len(zlab)):
    if (i%2 == 0):
        zlab[i] = ''
axs[0].set_xticklabels(xlab, fontsize=fs_tick)
axs[0].set_yticklabels(ylab, fontsize=fs_tick)
axs[0].set_zticklabels(zlab, fontsize=fs_tick)

# make the panes transparent
axs[0].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
axs[0].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
axs[0].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
axs[0].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
axs[0].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
axs[0].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

lw = 1.2
axs[0].plot([Ms[-1],Ms[-1]], [Ks[0],Ks[0]], [Vs[0],Vs[-1]], color='b', linewidth=lw, alpha=1.0, zorder=4)
axs[0].plot([Ms[-1],Ms[-1]], [Ks[0],Ks[-1]], [Vs[-1],Vs[-1]], color='b', linewidth=lw, alpha=1.0, zorder=5)
axs[0].plot([Ms[0],Ms[-1]], [Ks[0],Ks[0]], [Vs[-1],Vs[-1]], color='b', linewidth=lw, alpha=1.0, zorder=6)

###############
#### CBAR #####
###############
normalize = mcolors.Normalize(vmin=np.min(transmat_VM), vmax=np.max(transmat_VM))
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap='gray')
scalarmappaple.set_array(len(Vs))
cbar = fig.colorbar(scalarmappaple, cax=axs[4])
cbar.set_label('Success rate', fontsize=20)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.tight_layout()
plt.savefig('fig2.pdf', bbox_inches='tight')
plt.savefig('fig2.png', bbox_inches='tight')
plt.show()

# "On reconstruction of the Interferometric matrix"
# fig = plt.figure(figsize=(5,4))
# ax = fig.gca()
# plt.xlabel('M')
# plt.ylabel('Q')
# # im = ax.contourf(Qs, Ms, transmat, cmap='gray')
# im = ax.imshow(intM_transmat, cmap='gray')
# plt.locator_params(axis='x', nbins=len(Ms2))
# ax.set_xticks(np.arange(len(Ms2)))
# ax.set_xticklabels(Ms2)
# plt.locator_params(axis='y', nbins=len(Qs2))
# ax.set_yticks(np.arange(len(Qs2)))
# ax.set_yticklabels(Qs2)
# fig.colorbar(im, ax=ax)
# plt.show()