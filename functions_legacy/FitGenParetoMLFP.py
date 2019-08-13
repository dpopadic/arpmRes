import matplotlib.pyplot as plt
import numpy as np
from numpy import ptp, var, array, sort, abs, mean, log, exp
from numpy import sum as npsum, max as npmax

from scipy.optimize import fmin


def FitGenParetoMLFP(x, p):
    # This function finds the parameters csi and theta determining the
    # Generalized Pareto Distribution that best fits the dataset x with
    # Flexible Probabilities p.
    #  INPUTS
    #  x          :[row vector] dataset of invariants below the threshold
    #  p          :[row vector] Flexible Probabilities associated to x
    #  OPS
    #  csi_MLFP    :[scalar] first parameter of Generalized Pareto Distribution
    #  sigma_MLFP  :[scalar] second parameter of Generalized Pareto Distribution

    ## Code

    p = p.flatten()  # make sure p is a column vector

    if not isinstance(x, np.ndarray):
        raise ValueError('stats:gpfit:VectorRequired. X must be a vector')
    elif any(x <= 0):
        raise ValueError('stats:gpfit:BadData. The data in X must be positive')

    # if exist(OCTAVE_VERSION,builtin) == 0:
    #     options = statset(gpfit)
    # else:
    #     options = options(TolX,1.0000e-06, MaxFunEvals,400, MaxIter,200, Display,off)
    #
    #
    n = len(x)
    x = sort(x.flatten())
    xmax = x[-1]
    rangex = ptp(x)

    # # Can't make a fit.
    # if n == 0 or isfinite(rangex):
    #     csi_MLFP = zeros((1,1))
    #     sigma_MLFP = zeros((1,1))
    #     return
    # elif rangex < min(classX):
    #     # When all observations are equal, try to return something reasonable.
    #     if xmax <= sqrt(realmax(classX)):
    #         csi_MLFP = cast(zeros,classX)
    #         sigma_MLFP = cast(0,classX)
    #     else:
    #         csi_MLFP = cast(-Inf)
    #         sigma_MLFP = cast(Inf)
    #
    #     return
    # Otherwise the data are ok to fit GP distr, go on.

    # This initial guess is the method of moments:
    xbar = mean(x)
    s2 = var(x)
    k0 = -.5 * (xbar ** 2 / s2 - 1)
    sigma0 = .5 * xbar * (xbar ** 2 / s2 + 1)
    if k0 < 0 and (xmax >= -sigma0 / k0):
        # Method of moments failed, start with an exponential fit
        k0 = 0
        sigma0 = xbar

    parmhat = array([k0, log(sigma0)])

    # Maximize the log-likelihood with respect to k and lnsigma.
    # opts = {'maxiter':200}
    res = fmin(negloglike, parmhat, args=(x, p), maxiter=200)
    parmhat = res

    csi_MLFP = parmhat[0]
    sigma_MLFP = exp(parmhat[1])
    parmhat[1] = exp(parmhat[1])

    # if (err == 0):
    #     # fminsearch may print its own output text in any case give something
    #     # more statistical here, controllable via warning IDs.
    #     if output.funcCount >= options.MaxFunEvals:
    #         wmsg = 'Maximum likelihood estimation did not converge.  Function evaluation limit exceeded.'
    #     else:
    #         wmsg = 'Maximum likelihood estimation did not converge.  Iteration limit exceeded.'
    #     raise Warning(''.join(['stats:gpfit:IterOrEvalLimit.T',wmsg]))
    # elif (err < 0):
    #     raise ValueError('stats:gpfit:NoSolution. Unable to reach a maximum likelihood solution')

    # tolBnd = options.TolBnd
    # atBoundary = False
    # if (parmhat[0] < 0) and (xmax > -parmhat[1]/parmhat[0] - tolBnd):
    #     raise Warning('stats:gpfit:ConvergeoBoundary, Maximum likelihood has converged to a boundary point of the parameter space.'
    #                   '\n Confidence intervals and standard errors can not be computed reliably')
    #     atBoundary = True
    # elif (parmhat[0] <= -1/2):
    #     raise Warning('stats:gpfit:ConvergeoBoundary, Maximum likelihood has converged to an estimate of K < -1/2 \n'
    #                   'Confidence intervals and standard errors can not be computed reliably')
    #     atBoundary = True

    return csi_MLFP, sigma_MLFP


def negloglike(parms, data, FP):
    # Negative log-likelihood for the GP (log(sigma) parameterization).
    k = parms[0]
    lnsigma = parms[1]
    sigma = exp(lnsigma)

    n = len(data)
    z = data / sigma

    if abs(k) > np.finfo(float).eps:
        if k > 0 or npmax(z) < -1 / k:
            u = 1 + k * z
            lnu = log(u)
            nll = npsum(FP * (lnsigma + (1 + 1 / k) * lnu))
        else:
            # The support of the GP when k<0 is 0 < x < abs((sigma/k)).
            nll = np.inf
    else:  # limiting exponential dist.Tn as k->0
        nll = npsum(FP * (lnsigma + z))
    return nll
