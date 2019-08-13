import matplotlib.pyplot as plt
import numpy as np
from numpy import isnan, zeros, sort, where, argsort, sqrt, r_, concatenate
from numpy import sum as npsum
from numpy.linalg import pinv
from tqdm import trange

plt.style.use('seaborn')

from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
from MaxLikFPTReg import MaxLikFPTReg
from SmartInverse import SmartInverse


def DiffLengthMLFP(epsi, p, nu, threshold, smartinverse=0, maxiter=10 ** 5):
    # Max-Likelihood with Flexible Probabilities for Different-Length (DLFP) Series
    # INPUT
    #  epsi             : [matrix] (i_ x t_end) observations - with zeros's
    #  p                : [vector] (1 x t_end) flexible probabilities
    #  nu               : [scalar] degrees of freedom for the multivariate
    #                              Student t-distribution.
    #  threshold        : [scalar] or [vector](1 x 4) convergence thresholds.
    #  smartinverse      : [scalar] additional parameter: set it to 1 to use
    #                               LRD smart inverse in the regression process
    #  maxiter          : maximum number of iterations inside MaxLikFPTReg and MaxLikelihoodFPLocDispT
    # OP
    #  mu               : [vector] (i_ x 1) DLFP estimate of the location parameter.
    #
    #  sig2             : [matrix] (i_ x i_) DLFP estimate of the dispersion
    #                               parameter.
    #
    # --------------------------------------------------------------------------
    # NOTE:
    # 1) We suppose the missing values, if any, are at the beginning
    #   (the farthest observations in the past could be missing).
    #    We reshuffle the series in a nested pattern, such that the series with the
    #    longer history comes first and the one with the shorter history comes last

    # For details on the exercise, see here .

    ## Code
    if isinstance(threshold, float):
        threshold = [threshold, threshold, threshold, threshold]
    elif len(threshold) == 2 or len(threshold) == 3:
        threshold = [threshold[0], threshold[1], threshold[1], threshold[1]]

    i_, t_ = epsi.shape

    L = zeros(i_)
    for i in range(i_):
        L[i] = where(~np.isnan(epsi[i, :]))[0][0]

    Lsort, index = sort(L), argsort(L)
    epsi_sort = epsi[index, :]
    idx = argsort(index)

    c = 0
    epsi_nested = {}
    epsi_nested[c] = epsi_sort[[0]]
    t = zeros(i_, dtype=int)
    t[0] = Lsort[0]
    for j in range(1, i_):
        if Lsort[j] == Lsort[j - 1]:
            epsi_nested[c] = concatenate((epsi_nested[c], epsi_sort[[j], :]), axis=0)
        else:
            c = c + 1
            epsi_nested[c] = epsi_sort[[j]]
            t[c] = Lsort[j]
    # --------------------------------------------------------------------------

    c_ = len(epsi_nested)
    alpha, beta, s2 = {}, {}, {}
    alpha[0] = np.NaN
    beta[0] = np.NaN
    s2[0] = np.NaN

    # STEP 0: initialize
    mu, sig2, _ = MaxLikelihoodFPLocDispT(epsi_nested[0], p, nu, threshold[0], 1, smartinverse, maxiter)

    # STEP 1
    for c in range(1, c_):
        data = epsi_nested[c]
        data = data[:, ~isnan(data[0])]

        # a) probabilities
        p_k = p[[0], t[c]:t_] / npsum(p[[0], t[c]:t_])
        e = zeros((mu.shape[0], t_ - t[c]))
        sza = 1
        for j in range(c):
            szb = epsi_nested[j].shape[0]
            e[sza - 1:sza - 1 + szb, :] = epsi_nested[j][:, t[c]:t_]
            sza = sza + szb

        # b) degrees of freedom
        nu_c = nu + e.shape[0]

        # c) loadings
        alpha[c], beta[c], s2[c], _ = MaxLikFPTReg(data, e, p_k, nu_c, threshold[1:4], 1, smartinverse, maxiter)

        # d) location/scatter

        mu = mu.reshape(-1, 1)

        Mah = sqrt((e[:, [-1]] - mu).T.dot(pinv(sig2)) @ (e[:, [-1]] - mu)).squeeze()

        gamma = (nu_c / (nu + Mah)) * s2[c] + beta[c] @ sig2 @ beta[c].T
        sig2 = r_[r_['-1', sig2, sig2 @ beta[c].T], r_['-1', beta[c] @ sig2, gamma]]
        sig2 = (sig2 + sig2.T) / 2
        mu = r_[mu, alpha[c].reshape(-1, 1) + beta[c] @ mu]

    # STEP[1:] Output
    # reshuffling output
    mu = mu[idx]
    sig2 = sig2[np.ix_(idx, idx)]
    return mu, sig2
