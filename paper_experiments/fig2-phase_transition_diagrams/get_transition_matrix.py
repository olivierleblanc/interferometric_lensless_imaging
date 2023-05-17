import numpy as np

import sys, os
def updir(d, n):
  for _ in range(n):
    d = os.path.dirname(d)
  return d
sys.path.append(updir(__file__,2))

from list_utils import *

def get_transition_matrix(paths, thresh=40):
    path_inputs, path_GTnorms, path_errnorms, path_intM_GTnorms, path_intM_errnorms = paths
    inputs = np.array(read_list_from_txt(path_inputs))
    GTnorms = np.array(read_list_from_txt(path_GTnorms))
    errnorms = np.array(read_list_from_txt(path_errnorms))
    intM_GTnorms = np.array(read_list_from_txt(path_intM_GTnorms))
    intM_errnorms = np.array(read_list_from_txt(path_intM_errnorms))

    transmat = np.zeros((len(GTnorms), len(GTnorms[0])))
    intM_transmat = np.zeros(transmat.shape)
    for i,elem in enumerate(inputs):
        for j,pair in enumerate(elem):

            count = 0
            count2 = 0
            for k in range(len(GTnorms[i][j])):
                try:
                    count += 20*np.log10(GTnorms[i][j][k]/errnorms[i][j][k]) > thresh
                    if (intM_errnorms[i][j][k]>1e-20):
                        count2 += 20*np.log10(intM_GTnorms[i][j][k]/intM_errnorms[i][j][k]) > thresh
                except:
                    pass
            transmat[i,j] = count/len(GTnorms[i][j])
            intM_transmat[i,j] = count2/len(intM_GTnorms[i][j])
    return transmat, intM_transmat
