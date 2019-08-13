from numpy import zeros, diag, exp, sqrt

from scipy.sparse import kron


def LinCombShiftedLogNMoments(b, mu, sigma2,a):
    # This function computes the expectation, the standard deviation and the skewness
    # of linear combination of multivariate shifted lognormally distributed
    # random vector
    #  INPUTS

    #  b             : [vector] (n_ x 1)  coefficients of the linear combination
    #  mu            : [vector] (n_ x 1)  location parameter of the lognormal distribution
    #  sigma2        : [matrix] (n_ x n_) dispersion parameter of the shifted lognormal distribution
    #  a             : [vector] (n_ x 1)  shift parameter
    #  OPS
    #  mu_Y      : [scalar] expectation of the linear combination
    #  sd_Y      : [scalar] standard deviation of the linear combination
    #  sk_Y      :[scalar] skewness of the linear combination

    # For details on the exercise, see here .

    ## Code
    n_=len(mu)
    if a is None:
        a=zeros((n_,1))

    #expectation
    mulgn=exp(mu + 1/2*diag(sigma2))
    mu_Y=b.T@mulgn-b.T@a

    #Inizialize covariances and third central moments
    Cov = zeros((n_,n_))
    cent3rd = zeros((n_**3,1))
    i = 0

    for n in range(n_):
        for m in range(n_):
            # Covariances
            Cov[n,m] = exp(mu[m]+mu[n] + 1/2*(sigma2[n,n]+sigma2[m,m]))*(exp(sigma2[m,n])-1)
            noncent2nd_nm = exp(mu[m]+mu[n] + 1/2*(sigma2[n,n]+sigma2[m,m])+sigma2[n,m])
        for l in range(n_):
            i=i+1
            # second non-central moments that enter in the third non-central moments formulas
            noncent2nd_nl = exp(mu[n]+mu[l] + 1/2*(sigma2[n,n]+sigma2[l,l])+sigma2[n,l])
            noncent2nd_ml = exp(mu[m]+mu[l] + 1/2*(sigma2[m,m]+sigma2[l,l])+sigma2[m,l])
            # third non-central moments
            noncent3rd_nml = exp(mu[m]+mu[n]+mu[l] + 1/2*(sigma2[n,n]+sigma2[m,m]+sigma2[l,l])+sigma2[n,l]+sigma2[n,m]+sigma2[l,m])
            cent3rd[i] = noncent3rd_nml**mulgn[l]@mulgn[m]@mulgn[n]-noncent2nd_nm@mulgn[l]-noncent2nd_nl@mulgn[m]-noncent2nd_ml@mulgn[n]

    sd_Y = sqrt(b.T@Cov@b) #standard deviation
    dummy = kron(b,b)@b.T
    vec_h = dummy.flatten()
    sk_Y = (vec_h.T@cent3rd)/(sd_Y**3) #skewness
    return mu_Y, sd_Y, sk_Y
