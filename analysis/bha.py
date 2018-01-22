from __future__ import absolute_import, division, print_function
import os
import numpy as np

#from .due import due, Doi
#
#
## Use duecredit (duecredit.org) to provide a citation to relevant work to
## be cited. This does nothing, unless the user has duecredit installed,
## And calls this with duecredit (as in `python -m duecredit script.py`):
#due.cite(Doi("10.1038/srep10532"),
#         description="A novel brain partition highlights the modular skeleton shared by structure and function.",
#         tags=["reference-implementation"],
#         path='bha')


def cross_modularity(A, B, alpha, beta, T):
    """
    Given two input (symmetrical) matrices A and B, this function
    calculates the crossmodularity index X
    
    Parameters
    ----------
    A : array
        squared matrice of N*N (typically connectivity matrices), being N the number of ROIs
    B : array
        squared matrice of N*N (typically connectivity matrices), being N the number of ROIs
    alpha : float
        artibitrary thersholds to binarize the two  matrices (necessary for the similarity calculation)
    beta : float
        artibitrary thersholds to binarize the two  matrices (necessary for the similarity calculation)
    T : array
        label vector: each element vector is defined as an integer corresponding to the module that ROI belongs to
    
    Returns
    -------
    X : float
        crossmodularity
    Qa : array
        modularities of inA associatted to partition T
    Qb : array
        modularities of inB associatted to partition T
    L: float
        similarity between A and B

    """
    # Get the different labels of the modules
    labels = np.unique(T)

    # For each module compute sorensen index
    sorensen = np.zeros(len(labels))
    indx_m = np.empty(0)
    for m in labels:
        # Select the rois of each module and binarizes the resulting matrices using alpha and betha
        indx_m = np.array(np.where(T == labels[m]))
        indx = np.ix_(indx_m[0], indx_m[0])
        bin_A = A[indx] > alpha
        bin_B = B[indx] > beta
        bin_A = bin_A.astype(int)
        bin_B = bin_B.astype(int)
        sorensen[m] = np.sum(2*(np.multiply(bin_A, bin_B))) / (np.sum(bin_A) + np.sum(bin_B))
    # The total similarity is the mean similarity of all the modules
    L = np.mean(sorensen)

    # Compute the modularity index
    Qa = np.abs(modularity_index(np.absolute(A), T))
    Qb = np.abs(modularity_index(np.absolute(B), T))

    # Compute the cross modularity
    X = np.power((np.multiply(np.multiply(Qa, Qb), L)), 1/3)

    return X, Qa, Qb, L


def modularity_index(A, T):
    """
    A newman spectral algorithm adapted from the brain connectivity toolbox. 
    Original code:  https://sites.google.com/site/bctnet/measures/list 
    
    Parameters
    ----------
    A : array
        squared matrice of N*N (typically connectivity matrices), being N the number of ROIs
    T : array
        label vector: each element vector is defined as an integer corresponding to the module that ROI belongs to
    
    Returns
    -------
    Q : float
        modularity index
     """
    
    N = np.amax(np.shape(A))  # number of vertices
    K = np.sum(A, axis = 0, keepdims=True )  # degree
    
    m = np.sum(K)  # number of edges (each undirected edge is counted twice)
    B = A - np.divide(K.T.dot(K), m)  # modularity matrix
    
    if T.shape[0] == 1:
        T= T.T

    s = np.array([T,]*N).T #  compute modularity
    zero_idx = np.where((s - s.T)==0)
    others_idx = np.where((s - s.T)!=0)

    s[zero_idx] = 1
    s[others_idx] = 0
    
    Q = (s * B) / m
    Q = np.sum(Q)

    return Q

#
#if __name__ == "__main__":
#
#    from bha.utils import fetch_bha_data
#    from scipy import spatial, cluster
#
#    if not os.path.exists(os.path.join(data_path, 'average_networks.npz')):
#        fetch_bha_data()
#
#    data = np.load('bha/data/average_networks.npz')
#    struct_network = data.f.struct_network
#    func_network = data.f.func_network
#
#    # These parameters are based on the reference paper
#    num_clusters = 20
#    alpha = 0.45
#    beta = 0.0
#    struct_network = struct_network / np.max(struct_network)
#
#    """
#    Functional dendogram -> structure follows function
#    """
#
#    Y = spatial.distance.pdist(func_network, metric='cosine')
#    Z = cluster.hierarchy.linkage(Y, method='weighted')
#    T = cluster.hierarchy.cut_tree(Z, n_clusters=num_clusters)
#
#    Xsf, Qff, Qsf, Lsf = cross_modularity(func_network, struct_network,
#                                          alpha, beta, T[:, 0])
#
#    """
#    Structural dendogram  ->  function follows structure
#
#    X=1-struct_network
#    Y = zeros(1,size(X,1)*(size(X,1)-1)/2)
#    idxEnd=0
#    for i=1:size(X,1)-1
#        Y(idxEnd+1:idxEnd+length(X(i,i+1:end)))=X(i,i+1:end)
#        idxEnd=idxEnd+length(X(i,i+1:end))
#    end
#    Z = linkage(Y,'average')
#    H,T,permAll = dendrogram(Z,num_clusters,'colorthreshold',1000)
#    Xfs Qfs Qss Lfs =crossmodularity(func_network,struct_network,alpha,beta,T)
#    """
