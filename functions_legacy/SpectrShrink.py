import numpy as np
from numpy import histogram, ceil, mean, tile, r_
from numpy import sum as npsum, max as npmax, diagflat, diff, ones
from pcacov import pcacov
from scipy.interpolate import interp1d

from MarchenkoPastur import MarchenkoPastur


def SpectrShrink(sigma2_in,t_):
    # This function performs spectrum shrinkage on the input covariance matrix.
    #  INPUTS
    # sigma_in     :[matrix](i_ x i_) input estimated covariance matrix
    # t_end           :[scalar] length of the time series from which sigma2_in was estimated
    #  OPS
    # sigma2_out   :[matrix](i_ x i_) output shrunk covariance
    # lambda2_out  :[vector](i_ x 1) vector of shrunk eigenvalues
    # k_           :[scalar] optimal index separating the signals from noise
    # err          :[scalar] approximation error (distance) between M-P approximation and noise histogram
    # y_mp         :[vector](100 x 1) ordinates of M-P approximation of noise histogram
    # x_mp         :[vector](100 x 1) abscissas of M-P approximation of noise histogram
    # dist         :[vector] vector of approximation errors as a function of k

    # For details on the exercise, see here .

    ## Code
    i_ = sigma2_in.shape[0]

    # PCA decomposition

    e,lambda2 = pcacov(sigma2_in)

    # Determine optimal k_
    l = 1000
    h = 100

    dist = ones((1,i_))*np.NAN
    for k in range(int(ceil(0.02*i_)),int(ceil(0.75*i_))):

        lambda2_k = lambda2[k+1:]
        lambda2_noise = mean(lambda2_k)

        q = t_/(i_-k-1)
        x_tmp, mp_tmp, x_lim = MarchenkoPastur(q,l,lambda2_noise)# compute M-P on a very dense grid
        if q > 1:
            x_tmp = r_[0, x_lim[0], x_tmp]
            mp_tmp = r_[0, mp_tmp[0], mp_tmp]
        l_M = npmax(lambda2_k)
        if l_M > x_tmp[-1]:
            x_tmp = r_[x_tmp, x_lim[1], l_M]
            mp_tmp = r_[mp_tmp, 0, 0]

        hgram,x_bin = histogram(lambda2_k,h)# compute the histogram of eigenvalues
        d = x_bin[1]-x_bin[0]
        hgram = hgram/(d*(i_-k))# normalize histogram

        interp = interp1d(x_tmp,mp_tmp)# interpolation
        mp = interp(x_bin[:-1]+diff(x_bin)/2)

        dist[0,k] = npsum((mp-hgram)**2)*(i_-k-1)

    err_tmp,k_tmp = np.nanmin(dist),np.nanargmin(dist)
    k_ = k_tmp
    err = err_tmp

    # Isotropy
    lambda2_out = lambda2
    lambda2_noise = mean(lambda2[k_:i_])

    lambda2_out[k_:i_] = tile(lambda2_noise,(i_-k_,1)).flatten()# shrunk spectrum

    # Output

    sigma2_out = e@diagflat(lambda2_out)@e.T

    x_mp, y_mp, _ = MarchenkoPastur(t_/(i_-k_),100,lambda2_noise)# compute M-P on a very dense grid
    return sigma2_out,lambda2_out,k_,err,y_mp,x_mp,dist
