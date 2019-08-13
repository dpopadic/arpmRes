import numpy as np
from numpy import eye, empty
from numpy.linalg import matrix_rank, solve

from pcacov import pcacov

def ConditionalPC(sigma2, d):
    # This def computes the conditional principal directions/variances of a
    # symmetric positive definite matrix
    # INPUTS
    #   sigma2    : [matrix] (n_ x n_) symmetric positive definite matrix
    #   d         : [matrix] (k_ x n_) full rank linear constraints matrix
    # OUTPUTS
    #   lambda2_d : [vector] (n_ x 1) conditional principal variances
    #   e_d       : [matrix] (n_ x n_) conditional principal directions matrix

    # For details on the exercise, see here .
    ## Code

    # general settings
    n_ = sigma2.shape[0]
    m_ = n_-matrix_rank(d)
    lambda2_d = empty((n_, 1))
    e_d = empty((n_, n_))

    # 0. initialize constraints
    a_n = d

    for n in range(n_):

        # 1. orthogonal projection matrix

        p = eye(n_)-a_n.T.dot(solve(a_n@a_n.T, a_n))

        # 2. conditional dispersion matrix
        s2 = p@sigma2@p

        # 3. conditional principal directions/variances

        eigvec, eigval = pcacov(s2)
        e_d[:, n] = eigvec[:, 0]
        lambda2_d[n] = eigval[0]

        # 4. Update augmented constraints matrix
        if n+1 <= m_-1:
            a_n = np.r_[a_n, e_d[:, [n]].T.dot(sigma2)]
        elif n+1 >= m_ & n+1 <= n_-1:
            a_n = e_d[:, :n+1].T.dot(sigma2)

    return lambda2_d, e_d