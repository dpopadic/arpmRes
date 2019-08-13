## Prepare the environment

import matplotlib.pyplot as plt
import numpy as np
from Riccati import Riccati
from numpy import ones, zeros, diag, eye, log, sqrt, tile, r_, newaxis
from numpy import sum as npsum
from numpy.linalg import solve, det, pinv

from scipy.optimize import minimize

np.seterr(invalid='ignore')

plt.style.use('seaborn')


def DCCfit(ct, rho2):
    # (We should update the log-likelihood in order to account for a student-t
    #  distribution of time series (right now it's used a normal
    #  log-likelihood)
    # INPUTS
    # ct:     [matrix]   i_ x t_end initial time series with marginal student-t
    #                            distributions
    # rho[1:]   [matrix]   i_ x i_ target correlation matrix
    # OPS
    # epsi:   [matrix]   i_ x t_end time series of invariants with correlation matrix
    #                            rho2
    # a,b,c   [scalars]          parameters of the fitted DCC(1,1) model
    # q2_last [matrix]   i_x i_  last matrix Q estimated by DCC(1,1)
    # R                  i_ x i_ x t_end vector of matrices containing the time
    #                            dependent correlations
    # Function inspired by the Ucsd_garch package available at https://www.kevinsheppard.com/UCSD_GARCH
    ## Code

    i_, t_ = ct.shape

    tolcon = 1e-6
    options = {'disp': False}
    dccstarting = np.array([.01, .97])
    cons = ({'type': 'ineq', 'fun': lambda x, A, b: b - A.dot(x), 'args': (ones(dccstarting.shape), 1 - 2 * tolcon)})
    lb = zeros(dccstarting.shape) + 2 * tolcon
    ub = [None]*len(dccstarting)
    bounds = list(zip(lb, ub))
    res = minimize(dcc_mvgarch_likelihood, dccstarting, args=(ct.T, rho2, 1, 1), constraints=cons, bounds=bounds,
                   options=options)
    if res.status in [4,8]:
        dccstarting = np.array([.01, .98])
        res = minimize(dcc_mvgarch_likelihood, dccstarting, args=(ct.T, rho2, 1, 1), constraints=cons, bounds=bounds,
                       options=options)
    dccparameters = res.x
    _, R2t, _, Qt = dcc_mvgarch_likelihood(dccparameters, ct.T, rho2, 1, 1, fulloutput=True)
    q2_last = Qt[:, :, -1]
    R = R2t
    a = dccparameters[0]
    b = dccparameters[1]
    c = 1 - a - b
    rho = Riccati(eye(i_), rho2)
    epsi = zeros((i_, t_))
    for t in range(t_):
        Rt = Riccati(eye(i_), R2t[:, :, t])
        epsi[:, t] = rho.dot(pinv(Rt)) @ ct[:, t]
    return epsi, a, b, c, q2_last, R


def dcc_mvgarch_likelihood(params, stdresid, rho2, P, Q, fulloutput=False):
    # PURPOSE:
    #        Restricted likelihood for use in the DCC_MVGARCH estimation and
    #        returns the likelihood of the 2SQMLE estimates of the DCC parameters
    #
    # USAGE:
    #        [logL, Rt, likelihoods]=dcc_garch_likelihood(params, stdresid, P, Q)
    #
    # INPUTS:
    #    params      - A P+Q by 1 vector of parameters of the form [dccPparametersdccQparameters]
    #    stdresid    - A matrix, t x k of residuals standardized by their conditional standard deviation
    #    P           - The innovation order of the DCC Garch process
    #    Q           - The AR order of the DCC estimator
    #
    # OUTPUTS:
    #    logL        - The calculated Quasi-Likelihood
    #    Rt          - a k x k x t 3 dimensional array of conditional correlations
    #    likelihoods - a t by 1 vector of quasi likelihoods
    #
    #
    # COMMENTS:
    #
    #
    # Author: Kevin Sheppard
    # kevin.sheppard@economics.ox.ac.uk
    # Revision: 2    Date: 12/31/2001

    t1_, i1_ = stdresid.shape
    alpha = params[:P]
    beta = params[P:P + Q]
    sumA = npsum(alpha)
    sumB = npsum(beta)

    # First compute Qbar, the unconditional Correlation Matrix
    Qbar = rho2.copy()

    # Next compute Qt
    m = max(P, Q)
    Qt = zeros((i1_, i1_, t1_ + m))
    Rt = zeros((i1_, i1_, t1_ + m))
    Qt[:, :, :m] = tile(Qbar[...,newaxis], (1, 1, m))
    Rt[:, :, :m] = tile(Qbar[...,newaxis], (1, 1, m))
    logL = 0
    likelihoods = zeros((1, t1_ + m))
    # The stdresid have expected value 1  maybe but in the variances
    stdresid = r_[zeros((m, i1_)), stdresid]
    for j in range(m, t1_ + m):
        Qt[:, :, j] = Qbar * (1 - sumA - sumB)
        for i in range(P):
            Qt[:,:, j]=Qt[:,:, j]+alpha[i]*(stdresid[[j - i],:].T@stdresid[[j - i],:])
        for i in range(Q):
            Qt[:,:, j]=Qt[:,:, j]+beta[i]*Qt[:,:, j-i]
        Rt[:, :, j] = Qt[:, :, j]/(sqrt(diag(Qt[:, :, j]).reshape(-1,1))@sqrt(diag(Qt[:, :, j]).reshape(-1,1)).T)
        likelihoods[0, j] = log(det(Rt[:, :, j])) + stdresid[j, :] @ solve(Rt[:, :, j], eye(Rt.shape[0])) @ stdresid[j, :].T
        logL = logL + likelihoods[0, j]
    Qt = Qt[:, :, m: t1_ + m]
    Rt = Rt[:, :, m: t1_ + m]
    logL = (1 / 2) * logL
    likelihoods = (1 / 2) * likelihoods[0, m:t1_ + m]
    if np.isreal(logL):
        pass
    else:
        logL = 1e+8
    if logL == np.inf:
        logL = 1e+8
    if fulloutput:
        return logL, Rt, likelihoods, Qt
    else:
        return logL.flatten()/1e6