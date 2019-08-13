from numpy import zeros, sqrt
from numpy.random import rand
from numpy.linalg import pinv

def GramSchmidt(sigma2):
    # This def generates the Gram-Schmidt matrix g: g.T * sigma2 * g = eye(n_)
    #  INPUTS
    #   sigma2 : [matrix] (n_ x n_) symmetric and positive (semi)definite matrix
    #  OUTPUTS
    #   g      : [matrix] (n_ x n_) Gram-Schmidt matrix

    # For details on the exercise, see here .
    ## Code
    n_ = max(sigma2.shape)
    g = zeros((n_, n_))  # initialize

    # Step 0. Initialization
    a = rand(n_, n_)
    for n in range(n_):
        v_n = a[:, [n]]
        for m in range(n):
            # Step 1. Projection
            u_m = (g[:,[m]].T.dot(sigma2).dot(v_n))*g[:, [m]]

            # Step 2. Orthogonalization
            v_n = v_n - u_m

        # Setp 3. Normalization
        g[:, [n]] = v_n/sqrt(v_n.T.dot(sigma2).dot(v_n))
    return g
