from numpy import ones, zeros, diag, abs, diagflat
from numpy.linalg import norm, solve, pinv
from scipy.linalg import sqrtm


def MinimumTorsion(s2, max_niter=10000):
    # This funciton computes the minimum-torsion matrix for diversification analysis
    # see A. Meucci, A. Santangelo, R. Deguest - "Measuring Portfolio Diversification Based on Optimized Uncorrelated Factors" to appear (2013)
    # INPUT
    # Sigma [matrix]: n_xn_ covariance matrix of the risk factors
    # OP
    # t     [matrix]: n_xn_ minimum-torsion of the covariance matrix Sigma

    # For details on the exercise, see here .

    ## Code

    # Correlation matrix
    sigma = diag(s2)**(1/2)
    C = diagflat(1/sigma)@s2@diagflat(1/sigma)
    c = sqrtm(C) # Riccati root of C

    n_ = s2.shape[0]
    # initialize
    d = ones((1, n_))
    f = zeros(max_niter)
    for i in range(max_niter):
        U = diagflat(d)@c@c@diagflat(d)
        u = sqrtm(U)
        q = solve(u,diagflat(d)@c)
        d = diag(q@c)
        pi_ = diagflat(d)@q # perturbation
        f[i] = norm(c - pi_,ord='fro')
        if i > 1 and abs(f[i] - f[i-1])/f[i]/n_ <= 10**-8:
            f = f[:i]
            break
        elif i == max_niter and abs(f[max_niter] - f[max_niter-1])/f[max_niter]/n_ > 10**-8:
              print('number of max iterations reached: n_iter = %d' % max_niter)
    x = pi_.dot(pinv(c))
    t = diagflat(sigma)@x@diagflat(1/sigma)
    return t
