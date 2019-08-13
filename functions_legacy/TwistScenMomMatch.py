from numpy import tile, dot
from numpy.linalg import pinv

from FPmeancov import FPmeancov
from TransposeSquareRoot import TransposeSquareRoot


def TwistScenMomMatch(x, p, mu_, s2_, method='Riccati', d=None):
    # This def twists scenarios x to match arbitrary moments mu_ sigma2_
    #  INPUTS
    #   x      : [matrix] (n_ x j_) scenarios
    #   p      : [vector] (1 x j_) flexible probabilities
    #   mu_    : [vector] (n_ x 1) target means
    #   s2_    : [matrix] (n_ x n_) target covariances
    #   method : [string] Riccati (default), CPCA, PCA, LDL-Cholesky, Gram-Schmidt
    #   d      : [matrix] (k_ x n_) full rank constraints matrix for CPCA
    #  OUTPUTS
    #   x_     : [matrix] (n_ x j_) twisted scenarios

    # For details on the exercise, see here .
    ## Code

    # Step 1. Original moments
    mu_x, s2_x = FPmeancov(x, p)

    # Step 2. Transpose-square-root of s2_x
    r_x = TransposeSquareRoot(s2_x, method, d)

    # Step 3. Transpose-square-root of s2_
    r_ = TransposeSquareRoot(s2_, method, d)

    # Step 4. Twist factors
    b = dot(r_, pinv(r_x))

    # Step 5. Shift factors
    a = mu_ - b.dot(mu_x)

    # Step 6. Twisted scenarios
    x_ = tile(a, (1, x.shape[1])) + b.dot(x)

    return x_
