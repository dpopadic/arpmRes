from numpy import ones, zeros, diff, diag, eye, mean, tile, ix_, diagflat
from numpy import sum as npsum
from numpy.linalg import det

from kMeansClustering import kMeansClustering


def HomCl(c2,options):
    # This functions performs homogeneous clustering by using the method
    # specified in options
    #  INPUTS
    # c2       :[matrix](n_ x n_) starting correlation matrix
    # options  :[struct] structure containing options for clustering
    #  OPS
    # c2_hom   :[matrix](n_ x n_) homogenized correlation matrix
    # c2_clus  :[matrix](n_ x n_) correlation matrix partitioned into clusters, but not homogenized
    # i_clus   :[vector](n_ x 1) index vector such that c2_clus = c2(i_clus,i_clus)
    # l_clus   :[vector](k_+1 x 1) vector such that l_hom(k+1):l_hom[k] is the vector of entries in the k-th cluster
    # gamma    :[scalar] shrinkage coefficient to ensure positivity
    # c2_det   :[scalar] determizerost of matrix c2_hom

    # For details on the exercise, see here .

    ## Code

    # Clustering
    if options.method =='exogenous':
        i_clus = options.i_c
        l_clus = options.l_c
        c2_c = c2[ix_(i_clus,i_clus)]
    elif options.method == 'kmeans':
        k_ = options.k_
        c2_c,i_clus,l_clus,_ = kMeansClustering(c2,k_,0,0,0)

    # Homogenization
    k_ = len(l_clus)-1
    l_diff = diff(l_clus)
    c2_clus = zeros((len(i_clus),len(i_clus)))
    for k1 in range(k_):
        d = range(l_clus[k1],l_clus[k1+1])
        b = l_diff[k1]
        if b > 1:
            c2_tmp = npsum(c2_c[ix_(d,d)]-eye(b)) / (b*(b-1))
            c2_clus[ix_(d,d)] = c2_tmp*ones((b,b))
        for k2 in range(k1,k_):
            d1 = range(l_clus[k2],l_clus[k2+1])
            b1 = l_diff[k2]
            c2_tmp = mean(c2_c[ix_(d,d1)])
            c2_clus[ix_(d,d1)] = c2_tmp*ones((b,b1))
            c2_clus[ix_(d1,d)] = c2_tmp*ones((b1,b))
            c2_clus = c2_clus - diagflat(diag(c2_clus)) + diagflat(ones((1,len(i_clus))))# 1 on main diagonals

            # Positivity
            gamma = 1
            step = 0.005
            threshold = 10**-12

            c2_hom = c2_clus
            c2_det = det(c2_hom)
    while c2_det<threshold and gamma>0:
        gamma = gamma-step
        c2_hom = gamma*c2_hom
        c2_hom = c2_hom - diagflat(diag(c2_hom)) + diagflat(ones((1,len(i_clus))))# 1 on main diagonals
        c2_det = det(c2_hom)

    if gamma==0:
        print('convergence not reached')
    return c2_hom, c2_clus, i_clus, l_clus, gamma, c2_det
