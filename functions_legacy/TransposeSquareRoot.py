from numpy import eye, diag, sqrt, diagflat
from numpy.linalg import cholesky, inv, solve

from ConditionalPC import ConditionalPC
from GramSchmidt import GramSchmidt
from Riccati import Riccati
from pcacov import pcacov


def TransposeSquareRoot(sigma2, method='Riccati', d=None):
    '''This def computes the transpose-square-root matrix s of a symmetric
    and positive (semi)definite matrix sigma2 such that sigma2 = s*s.T
     INPUTS
      sigma2 : [matrix] (n_ x n_) positive definite matrix
      method : [string] Riccati (default), CPCA, PCA, LDL-Cholesky,
      Gram-Schmidt, Chol
      d      : [matrix] (k_ x n_) full rank constraints matrix for CPCA
     OUTPUTS
      s      : [matrix] (n_ x n_) transpose-square-root of sigma2
    '''

    n_ = max(sigma2.shape)
    if method == 'Riccati':
        s = Riccati(eye(n_), sigma2)
    elif method == 'CPCA':
        lambda2_d, e_d = ConditionalPC(sigma2, d)
        s = solve(e_d.T, diagflat(sqrt(lambda2_d)))
    elif method == 'PCA':
        e, lambda2 = pcacov(sigma2)
        s = e.dot(diag(sqrt(lambda2))).dot(e.T)
    # elif method == 'LDL-Cholesky':
    #     l, delta_2 = ldl(sigma2)
    #     s = l*sqrt(delta_2)
    elif method == 'Gram-Schmidt':
        g = GramSchmidt(sigma2)
        s = inv(g).T
    elif method == 'Chol':
        s = cholesky(sigma2)
    return s
