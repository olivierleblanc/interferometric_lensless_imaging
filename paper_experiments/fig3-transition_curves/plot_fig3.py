import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import sys, os
os.chdir(os.path.dirname(__file__)) # go to current file directory

plt.rcParams['text.usetex'] = True

folder = 'simu_results/transition_curves/'
gathered_data = np.load(os.path.join(folder, 'gathered_data.npy'))
Ks = np.arange(5, 12)
Ms = np.arange(2, 138, 8)

fig = plt.figure(figsize=(10,6))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'$M$', fontsize=28)
plt.ylabel(r'Success rate', fontsize=28)

normalize = mcolors.Normalize(vmin=Ks[0], vmax=Ks[-1])
colormap = cm.jet
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(len(Ks))
cbar = fig.colorbar(scalarmappaple, ax=fig.gca())
cbar.set_label(r'$K$', fontsize=28)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
# cbar.ax.set_yticklabels(['3.0','','','','2.0','','','','1.0'])

for i,K in enumerate(Ks):
    plt.plot(Ms, gathered_data[i,:], color=colormap(normalize(K)), label='K={}'.format(str(K)))
plt.tight_layout()
plt.savefig('fig3.pdf')
plt.savefig('fig3.png')
plt.show()