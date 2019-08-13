import matplotlib.pyplot as plt
from numpy import array, ones, zeros, cov

plt.style.use('seaborn')
from fglasso import glasso


def GraphicalLasso(pop, lam, initStruct=None, approximate=0, warmInit=0, verbose=0, penalDiag=1, tolThreshold=1e-4, maxIter=1e4, w=None, theta=None):
    # [w, theta, iter, avgTol, hasError] = GraphicalLasso(pop, lam,
    # initStruct, approximate, warmInit, verbose, penalDiag, tolThreshold,
    # maxIter, w, theta)
    #
    # Computes a regularized estimate of covariance matrix and its inverse
    # Inputs:
    #  - pop: the set of samples to be used for covariance estimation in an NxP
    #  matrix where N is the number of samples and P is the number of variables
    #  - lam: the regularization penalty. Can be a single number or a matrix
    #  of penalization values for each entry in the covariance matrix
    #  - initStruct(o@): a matrix of size PxP, where zero entries will force
    #  the corresponding entries in the inverse covariance matrix to be zero.
    #  - approximate([o]): a flag indicating whether to use approximate estimation
    #  (Meinhausen-Buhlmann approximation)
    #  - warmInit[o]: a flag indicating whether the estimation will start from
    #  given initial values of w and theta
    #  - verbose([o]): a flag indicating whether to output algorithm process
    #  - penalDiag[o]: a flag indicating whether to penalize diagonal elements of
    #  the covariance matrix
    #  - tolThreshold[o]: the amount of tolerance acceptable for covariance matrix
    #  elements before terminating the algorithm
    #  - maxIter([o]): maximum number of iteration to perform in the algorithm
    #  - w[o]: the initial value of covariance matrix used for warm initialization
    #  - theta([o]): the initial value of inverse covariance matrix used for warm
    #  initialization
    # @: o indicates optional arguments
    # Outputs:
    #  - w: the estimated covariance matrix
    #  - theta: the estimated inverse covariance matrix
    #  - iter: actual number of iterations performed in the algorithm
    #  - avgTol: average tolerance of covariance matrix entries before
    #  terminating the algorithm
    #  - hasError: a flag indicating whether the algorithm terminated
    #  erroneously or not
    #
    # Code by: Hossein Karshenas (hkarshenas@fi.upm.es)
    # Date: 10 Feb 2011

    numVars = pop.shape[1]
    if isinstance(lam,float):
        lam = array([[lam]])
    m, n = lam.shape
    if m != n:
        raise ValueError('Regularization coefficients matrix should be symmetric matrix')
    elif m > 1 and m != numVars:
        raise ValueError('Regularization coefficients matrix should have a size equal to the number of variables')
    if m == 1:
        lam = lam * ones((numVars,numVars))

    if initStruct is not None:
        initStruct = 1 - initStruct
        initStruct = 10e9*initStruct
        lam = lam + initStruct

    if w is None:
        if warmInit is False:
            raise ValueError('In warm initialization mode starting values for the covariance and precision matrices should be determined')
        else:
            w = zeros((numVars,numVars),order='F')
    if theta is None:
        if warmInit is False:
            raise ValueError('In warm initialization mode starting values for the precision matrix should be determined')
        else:
            theta = zeros((numVars,numVars),order='F')

    niter = 0
    jerr = 0.0
    # glasso(cov(pop.T), lam, 1, 1, 1, penalDiag, tolThreshold, maxIter, w, theta, niter, jerr, 1)
    glasso(cov(pop.T), lam, approximate, 0, verbose, penalDiag, tolThreshold, maxIter, w, theta, niter, jerr, numVars)
    # w, theta, iter, avgTol, hasError = glasso(cov(pop.T), lam, approximate, 0, verbose, penalDiag, tolThreshold, maxIter, w, theta, 0, 0, numVars)

    if False:
        raise Warning('The execution of the algorithm caused errors')
    return w, theta, niter, jerr

