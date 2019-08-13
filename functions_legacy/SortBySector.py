from numpy import zeros, cumsum, r_

from SectorSelect import SectorSelect


def SortBySector(sectors,sector_names):
    # This function sorts the vector "sectors" according to the names contained
    # into vector "sector_names"
    #  INPUTS
    # sectors         :[char array](n_ x p_) array whose rows contain a list of sectors
    # sector_names    :[char array](k_ x p_) array whose rows contain the sector names
    #  OPS
    # index           :[vector](n_ x 1) vector of indeces such that sectors[index] is the sorted vectos
    # L               :[vector](1 x (n_+1)) vector such that index(L[i]+1:L(i+1)) are the indeces of companies in sector_names([i])

    ## Code
    n_ = sectors.shape[0]
    k_ = sector_names.shape[0]
    index_sectors = zeros((n_,1),dtype=int)

    index_tmp = SectorSelect(sectors, sector_names[0])
    L_tmp = zeros((1,k_),dtype=int)
    L_tmp[0,0] = len(index_tmp)
    L = cumsum(L_tmp)
    index_sectors[:L[0],0] = index_tmp

    for i in range(1,k_):
        index_tmp = SectorSelect(sectors, sector_names[i])
        L_tmp[0,i] = len(index_tmp)
        L = cumsum(L_tmp)
        index_sectors[L[i-1]:L[i],0] = index_tmp

    L = r_[0, L]
    return index_sectors, L
