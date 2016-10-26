"""
domain adaptation with optimal transport
"""
import numpy as np
from scipy.spatial.distance import cdist

from bregman import sinkhorn



def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def sinkhorn_lpl1_mm(a,labels_a, b, M, reg, eta=0.1):
    p=0.5
    epsilon = 1e-3

    # init data
    Nini = len(a)
    Nfin = len(b)
     
    indices_labels = []
    idx_begin = np.min(labels_a)
    for c in range(idx_begin,np.max(labels_a)+1):
        idxc = indices(labels_a, lambda x: x==c)
        indices_labels.append(idxc)

    W=np.zeros(M.shape)

    for cpt in range(10):
        Mreg = M + eta*W
        transp=sinkhorn(a,b,Mreg,reg,numItermax = 200)
        # the transport has been computed. Check if classes are really separated
        W = np.ones((Nini,Nfin))
        for t in range(Nfin):
            column = transp[:,t]
            all_maj = []
            for c in range(idx_begin,np.max(labels_a)+1):
                col_c = column[indices_labels[c-idx_begin]]
                if c!=-1:
                    maj = p*((sum(col_c)+epsilon)**(p-1))
                    W[indices_labels[c-idx_begin],t]=maj
                    all_maj.append(maj)

            # now we majorize the unlabelled by the min of the majorizations
            # do it only for unlabbled data
            if idx_begin==-1:
                W[indices_labels[0],t]=np.min(all_maj)
    
    return transp
    