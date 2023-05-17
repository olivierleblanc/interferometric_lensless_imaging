import matplotlib.pyplot as plt
import numpy as np

import sys, os
def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
os.chdir(updir(__file__,2))

plt.rcParams.update({
    "text.usetex": True, # LaTeX rendering
    "font.family": 'DejaVu Sans'
})
fs = 20 # font size for axis labels
fs_tick = 15 # font size for ticks

snr = lambda ref, rec : 20*np.log10(np.linalg.norm(ref)/np.linalg.norm(rec-ref))

folder = r"C:\Users\leblanco.OASIS\Documents\IngeCivilPHD\Work\LECI\Interferometric_LE\code\Exp_data_analysis\reconstruction_data\snr_vs_M_220624B/"

wi = 0.2
h = 0.425
py2 = 0.525
x0 = 0.04
w0 = 0.35

fig = plt.figure(figsize=(15,7))
axs = [ fig.add_axes([x0, 0.1, w0, 0.85]), 
[fig.add_axes([x0+w0, py2, wi, h]), fig.add_axes([x0+w0+wi, py2, wi, h]), fig.add_axes([x0+w0+2*wi, py2, wi, h])], 
[fig.add_axes([x0+w0, 0.1, wi, h]), fig.add_axes([x0+w0+wi, 0.1, wi, h]), fig.add_axes([x0+w0+2*wi, 0.1, wi, h])]]

#########################################
######## 1. SNR vs M ####################
#########################################
for i, line in enumerate(axs[1:]):
    for j, ax in enumerate(line):
        ax.set_xticks([])
        ax.set_yticks([])

paths = [ f.path for f in os.scandir(folder) if (f.path.split('/')[-1][0]!='.')]
paths = [path for path in paths if (path.split('Q')[-1][0]=='1' and path[-3:]=='npz' and ('N256' in path)) ]

Ms = np.zeros(len(paths))
snrs_m = np.zeros(len(paths))
snrs_std = np.zeros(len(paths))
W = np.load(paths[0])['arr_1'].shape[0]
sols = np.zeros((len(paths), W, W))

for i, path in enumerate(paths):
  data = np.load(path)
  Ms[i] = data['arr_0']
  sols[i] = data['arr_1']
  snrs_m[i] = np.mean(data['arr_2'])
  snrs_std[i] = np.std(data['arr_2'])

inds = np.argsort(Ms)
Ms = Ms[inds]
snrs_m = snrs_m[inds]
snrs_std = snrs_std[inds]

paths2 = [ f.path for f in os.scandir(folder) if (f.path.split('/')[-1][0]!='.')]
paths2 = [path for path in paths2 if (path.split('Q')[-1][0]=='5' and path[-3:]=='npz' and ('N256' in path)) ]

Ms2 = np.zeros(len(paths2))
snrs_m2 = np.zeros(len(paths2))
snrs_std2 = np.zeros(len(paths2))
W2 = np.load(paths2[0])['arr_1'].shape[0]
sols2 = np.zeros((len(paths2), W2, W2))

for i, path in enumerate(paths2):
  data = np.load(path)
  Ms2[i] = data['arr_0']
  sols2[i] = data['arr_1']
  snrs_m2[i] = np.mean(data['arr_2'])
  snrs_std2[i] = np.std(data['arr_2'])

inds2 = np.argsort(Ms2)
Ms2 = Ms2[inds2]
snrs_m2 = snrs_m2[inds2]
snrs_std2 = snrs_std2[inds2]

K = len(data['arr_2']) # number of folds
print("K = {}".format(K))

axs[0].set_xlabel(r'$M$', fontsize=fs)
axs[0].set_ylabel('SNR', fontsize=fs)
axs[0].plot(Ms, snrs_m, 'r.-', linewidth=2.0, label=r'$Q=110$')
axs[0].fill_between(Ms, snrs_m-snrs_std, snrs_m+snrs_std, alpha=0.15, color='r')
axs[0].plot(Ms2, snrs_m2, 'b.-', linewidth=2.0, label=r'$Q=55$')
axs[0].fill_between(Ms2, snrs_m2-snrs_std2, snrs_m2+snrs_std2, alpha=0.15, color='b')
axs[0].legend(loc='lower right', fontsize=18)
axs[0].tick_params(axis='both', which='major', labelsize=fs_tick)

#########################################
############## 2. GT ####################
#########################################
low=50
f_ds = np.load(folder+'GT.npy')
axs[1][0].imshow(f_ds[low:-low,low:-low])

#########################################
############# 3. RS rec #################
#########################################
RS = np.load(r'C:\Users\leblanco.OASIS\Documents\IngeCivilPHD\Work\LECI\Interferometric_LE\code\Figures_for_papers\figs\RS_rec.npy')
axs[2][0].imshow(RS[low:-low,low:-low])

#########################################
########### 4. MCF-LI rec ###############
#########################################
axs[1][1].imshow(sols2[0][low:-low,low:-low])
axs[2][1].imshow(sols[0][low:-low,low:-low])

axs[1][2].imshow(sols2[-1][low:-low,low:-low])
axs[2][2].imshow(sols[-1][low:-low,low:-low])

thesplit = paths2[0].split('_M')
thesplit[0] = thesplit[0]+'_M'
thesplit[-1] = thesplit[-1][3:-4]
thesplit = ''.join(thesplit)
# plt.savefig(r'C:\Users\leblanco.OASIS\Documents\IngeCivilPHD\Work\LECI\Interferometric_LE\papers\IEEE_TCI_journal\images\exp_results.pdf', bbox_inches='tight')
plt.savefig(r'C:\Users\leblanco.OASIS\Documents\IngeCivilPHD\Work\LECI\Interferometric_LE\talks\Euler23\images\exp_results.png', bbox_inches='tight')
plt.show()