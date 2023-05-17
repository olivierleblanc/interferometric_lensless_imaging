"""
    Author : Olivier Leblanc
    Date : 13/01/2023

    Code description :
    __________________
    For many number of cores Q, draw a random set \Omega of core positions in [0,N] space with N=256, compute the set of position differences \Omega-\Omega, and compute the cardinality of the set of unique frequencies. 
"""

import numpy as np
from matplotlib import pyplot as plt

"________________________________________________"

# Qs = np.linspace(1,64,20).astype('int')
Qs = np.arange(2,120)
N = 256
ntrial = 150

eff_visibility_cardinality = np.zeros((Qs.shape[0], ntrial))

for i,Q in enumerate(Qs):
    for trial in range(ntrial):
        pos_cores = np.random.permutation(np.arange(N//2))[:Q] # random cores locations
        # pos_cores = np.round(np.arange(Q)*(N-1)/(Q-1)).astype(int) # regularly spaced cores locations

        "Define Om = {p_j - p_k, j,k \in [Q]}"
        Om = np.subtract.outer(pos_cores, pos_cores).astype(int)
        Om_up = Om[np.triu_indices(Om.shape[0],k=1)]
        unique_Om = np.unique(Om_up)
        eff_visibility_cardinality[i,trial] = unique_Om.shape[0]
        # print('There are {} unique frequencies compared to the maximum possible value Q(Q-1)/2={}'.format(unique_Om.shape[0], int(Q*(Q-1)/2)))

mean_eff_visibility_cardinality = np.mean(eff_visibility_cardinality, axis=1)
std_eff_visibility_cardinality = np.std(eff_visibility_cardinality, axis=1)

Q_vis_bijection = np.array([Qs, mean_eff_visibility_cardinality]).T 


plt.figure()
plt.plot(Qs, mean_eff_visibility_cardinality, 'r')
plt.fill_between(Qs, mean_eff_visibility_cardinality-std_eff_visibility_cardinality, mean_eff_visibility_cardinality+std_eff_visibility_cardinality, alpha=0.2, color='r')
# plt.plot(Qs*(Qs-1), mean_eff_visibility_cardinality, 'r')
# plt.fill_between(Qs*(Qs-1), mean_eff_visibility_cardinality-std_eff_visibility_cardinality, mean_eff_visibility_cardinality+std_eff_visibility_cardinality, alpha=0.2, color='r')
plt.xlabel(r'$Q$')
# plt.xlabel(r'$Q(Q-1)$')
plt.ylabel(r'$|\mathcal{V}|$')
# plt.xlim([0,150])
plt.show()