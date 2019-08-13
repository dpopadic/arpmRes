from numpy import argmax, zeros, argsort, cumsum, floor, mean, arange, histogram, ix_
from numpy import sum as npsum

from sklearn.cluster import KMeans

from NormalScenarios import NormalScenarios


def kMeansClustering(c2,k_,i_s,l_s,opt=0):
    # This function performs k-means clustering
    #  INPUTS:
    # c2          :[matrix](n_ x n_) starting correlation matrix
    # k_          :[scalar] max number of clusters
    # i_s         :[vector](n_ x 1) index vector such that i_s((l_s[i]+1:l_s(i+1))) points to the companies in the i-th sector
    # l_s         :[vector]
    # opt         :[scalar]if opt==1 the function performs k-means clustering for every k<=k_
    #  OPS:
    # c2_c        :[matrix](n_ x n_) correlation matrix sorted into clusters
    # i_c         :[vector](n_ x 1) index vector such that c2_c = c2(i_c,i_c)
    # l_c         :[vector]
    # sect2clust  :[vector](k_ x 1) vector of indeces that establish a correspondance between the k_ sectors and the k_ clusters

    ## Code

    #generate (n_ x t_end) sample with target corr = rho2 and mean = 0
    Model = 'Riccati'
    n_ = len(c2)
    j_ = int(floor(n_*2))
    mu = zeros((n_,1))
    Epsi = NormalScenarios(mu,c2,j_,Model)[0]

    n_sector = len(l_s)-1

    if opt == 1:
        i_c = zeros((n_,k_))
        l_c = zeros((k_+1,k_))
        c2_c = zeros((n_,n_,k_))

        for k in range(k_):
            # find k clusters
            km = KMeans(n_clusters=k)
            if k == n_sector:
                C_start = zeros((k_,j_))
                for k1 in range(k_):
                    C_start[k1] = mean(Epsi[i_s[l_s[k1]:l_s[k1+1]],:])
                km.fit(Epsi)
            else:
                km.fit(Epsi)
            IDX, mu = km.labels_, km.cluster_centers_
            #sort by clusters
            i_c[:,k] = argsort(IDX)
            l_tmp = zeros((k,1))
            for i in range(k):
                l_tmp[i] = npsum(IDX == i)
            l_c[:k+1,k] = [0, cumsum(l_tmp)]
            c2_c[:,:,k] = c2[ix_(i_c[:,k],i_c[:,k])]
    else:
            # find k_ clusters
            km = KMeans(n_clusters=k_)
            if k_ == n_sector:
                C_start = zeros((k_,j_))
                for k1 in range(k_):
                    C_start[k1] = mean(Epsi[i_s[l_s[k1]:l_s[k1+1]],:])
                km.fit(Epsi)
            else:
                km.fit(Epsi)
            IDX, mu = km.labels_, km.cluster_centers_
            #sort by clusters
            i_c = argsort(IDX)
            l_tmp = zeros((k_,1))
            for i in range(k_):
                l_tmp[i] = npsum(IDX == i)
            l_c = [0, cumsum(l_tmp)]
            c2_c = c2[ix_(i_c,i_c)]

    #find agreement between sectors and clusters
    sect2clust = zeros((k_,1))
    p = zeros((k_,n_sector))
    for k in range(n_sector):
        id = i_s[l_s[k]:l_s[k+1]] # comps in sector k
        p[:,k] = histogram(IDX[id].T,bins=arange(1,k_+2))[0]#joint probability
    for k in range(k_):
        sect2clust[k] = argmax(p[k])

    return c2_c, i_c, l_c, sect2clust
