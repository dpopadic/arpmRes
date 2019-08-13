import numpy as np
from numpy import sum as npsum
from numpy import zeros, tile, r_, squeeze
from numpy.linalg import solve, norm

from SmartInverse import SmartInverse


def MaxLikelihoodFPLocDispT(epsi, p, nu, threshold, last=0, smartinverse=0, maxiter=10 ** 5):
    # This function estimates the Maximum Likelihood with Flexible Probabilities location and dispersion parameters of the invariants under the Student t distribution assumption by
    # means of an iterative algorithm (MaxLikelihoodFPLocDispT routine)
    #  INPUT
    #  epsi            :[matrix](i_ x t_end) timeseries of invariants
    #   p              :[vector](1 x t_end) flexible probabilities associated to the invariants
    #  nu              :[scalar] degrees of freedom of the t-Student distribution
    #  threshold       :[scalar] or [vector](1 x 2) convergence threshold
    #  last  (optional):[scalar] if last!=0 only the last computed mean and covariance are returned
    #  maxiter         :[scalar] maximum number of iterations
    #  OUTPUT
    #  mu_MLFP      :[matrix](i_ x k_) array containing the mean vectors computed at each iteration
    #  sigma2_MLFP  :[array](i_ x i_ x k_) array containing the covariance matrices computed at each iteration
    #  error        :[vector](2 x 1) vector containing the relative errors at the last iteration

    # For details on the exercise, see here .

    ## Code

    if isinstance(threshold, float):
        threshold = r_[threshold, threshold]

    # initialize
    i_, t_ = epsi.shape
    mu_MLFP = zeros((i_, 1))
    sigma2_MLFP = zeros((i_, i_, 1))

    mu_MLFP[:, [0]] = epsi @ p.T
    epsi_c = epsi - tile(mu_MLFP[:, [0]], (1, t_))

    sigma2_MLFP[:, :, 0] = epsi_c @ (np.diagflat(p) @ epsi_c.T)

    error = [10 ** 6, 10 ** 6]
    k = 0
    while npsum(error > threshold) >= 1 and k < maxiter:
        k = k + 1
        # update weigths
        epsi_c = epsi - tile(mu_MLFP[:, [k - 1]], (1, t_))

        w_den = nu + npsum(epsi_c * solve(sigma2_MLFP[:, :, k - 1], epsi_c), 0)
        w = (nu + i_) / w_den
        # update output
        mu_MLFP = r_['-1', mu_MLFP, (npsum(tile(p * w, (i_, 1)) * epsi, 1) / (p @ w.T))[..., np.newaxis]]
        epsi_c = epsi - tile(mu_MLFP[:, [k]], (1, t_))
        sigma2_MLFP = r_['-1', sigma2_MLFP, (epsi_c @ np.diagflat(p * w) @ epsi_c.T)[..., np.newaxis]]
        sigma2_MLFP[:, :, k] = (squeeze(sigma2_MLFP[:, :, k]) + squeeze(sigma2_MLFP[:, :, k]).T) / 2
        # convergence
        error[0] = norm(mu_MLFP[:, k] - mu_MLFP[:, k - 1]) / norm(mu_MLFP[:, k])
        error[1] = norm(sigma2_MLFP[:, :, k] - sigma2_MLFP[:, :, k - 1], ord='fro') / norm(sigma2_MLFP[:, :, k],
                                                                                           ord='fro')

    if last != 0:
        mu_MLFP = mu_MLFP[:, -1]
        sigma2_MLFP = sigma2_MLFP[:, :, -1]

    return mu_MLFP, sigma2_MLFP, error
