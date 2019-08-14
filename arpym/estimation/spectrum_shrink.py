# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from arpym.statistics.marchenko_pastur import marchenko_pastur
from arpym.tools.pca_cov import pca_cov


def spectrum_shrink(sigma2_in, t_):
    """For details, see here.

    Parameters
    ----------
         sigma_in : array, shape (i_, i_)
         t_ : scalar

    Returns
    -------
         sigma_out : array, shape (i_, i_)
         lambda2_out : array, shape (i_, )
         k_ : scalar
         err : scalar
         y_mp : array, shape (100, )
         x_mp : array, shape (100, )
         dist : array

    """

    i_ = sigma2_in.shape[0]

    # PCA decomposition

    e, lambda2 = pca_cov(sigma2_in)

    # Determine optimal k_
    ll = 1000

    dist = np.ones(i_-1)*np.nan

    for k in range(i_-1):

        lambda2_k = lambda2[k+1:]
        lambda2_noise = np.mean(lambda2_k)

        q = t_/len(lambda2_k)

        # compute M-P on a very dense grid
        x_tmp, mp_tmp, x_lim = marchenko_pastur(q, ll, lambda2_noise)
        if q > 1:
            x_tmp = np.r_[0, x_lim[0], x_tmp]
            mp_tmp = np.r_[0, mp_tmp[0], mp_tmp]
        l_max = np.max(lambda2_k)
        if l_max > x_tmp[-1]:
            x_tmp = np.r_[x_tmp, x_lim[1], l_max]
            mp_tmp = np.r_[mp_tmp, 0, 0]

        # compute the histogram of eigenvalues
        hgram, x_bin = np.histogram(lambda2_k, len(x_tmp), density=True)

        # interpolation
        interp = interp1d(x_tmp, mp_tmp, fill_value='extrapolate')
        mp = interp(x_bin[:-1])

        dist[k] = np.mean((mp-hgram)**2)

    err_tmp, k_tmp = np.nanmin(dist), np.nanargmin(dist)
    k_ = k_tmp
    err = err_tmp

    # Isotropy
    lambda2_out = lambda2
    lambda2_noise = np.mean(lambda2[k_+1:])

    lambda2_out[k_+1:] = lambda2_noise  # shrunk spectrum

    # Output

    sigma2_out = e@np.diagflat(lambda2_out)@e.T

    # compute M-P on a very dense grid
    x_mp, y_mp, _ = marchenko_pastur(t_/(i_-k_-1), 100, lambda2_noise)

    return sigma2_out, lambda2_out, k_, err, y_mp, x_mp, dist
