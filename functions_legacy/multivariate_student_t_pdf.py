import numpy as np
from numpy import log, diag, exp
from numpy import sum as npsum
from numpy.linalg import cholesky as cholesky
from numpy.linalg import solve as solve
from scipy.special import gammaln as gammaln


def multivariate_student_t_pdf(x, mu, Sigma2, df):
    x = np.atleast_2d(x)  # requires x as 2d
    n_ = Sigma2.shape[0]  # dimensionality

    R = cholesky(Sigma2)

    z = solve(R, x)

    logSqrtDetC = npsum(log(diag(R)))
    logNumer = -((df + n_) / 2) * log(1 + npsum(z ** 2, axis=0) / df)
    logDenom = logSqrtDetC + (n_ / 2) * log(df * np.pi)

    y = exp(gammaln((df + n_) / 2) - gammaln(df / 2) + logNumer - logDenom)

    return y