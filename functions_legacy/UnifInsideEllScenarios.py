from numpy import sum as npsum
from numpy import zeros, diag, eye, sqrt, tile, diagflat
from numpy.linalg import cholesky, solve
from numpy.random import rand

from NormalScenarios import NormalScenarios
from DimRedScenariosNormal import DimRedScenariosNormal


def UnifInsideEllScenarios(mu,sig2,j_,k_=0,method='Riccati',d=None):
    # This function generates j_ Monte carlo scenarios of an elliptical random variable
    # with location vector mu, dispersion matrix sigma2 and radial component
    # with quantile function q_R
    #  INPUTS
    #   mu      : [vector]            (n_ x 1) location vector
    #   sigma2  : [matrix]            (n_ x n_) dispersion matrix
    #   j_      : [scalar]            number of simulations
    #   k_      : [scalar]           (optional) number of factors for dimension
    #                                 reduction (we advise to take k_<<n_)
    # method    : [string]            Riccati (default), CPCA, PCA, LDL-Cholesky, Gram-Schmidt
    # d         : [matrix]           (k_ x n_) full rank constraints matrix for CPCA
    #  OPS
    #   X       : [matrix]           (n_ x j_) matrix of elliptical simulations
    #   R       : [vector]           (1 x j_)  vector of radial component scenarios
    #   Y       : [matrix]           (n_ x j_) matrix of uniform component scenarios

    # For details on the exercise, see here .
    ## Code

    n_ = len(mu) # number of variables

    # Step 1. Radial Scenarios
    R=(rand(1,j_))**(1/n_)

    # Step 2. Correlation
    rho2=diag(diag(sig2)**(-1/2))@sig2@diag(diag(sig2)**(-1/2))

    # Step 3. Normal scenarios
    if k_==0:
       N=NormalScenarios(zeros((n_,1)),rho2,j_,'Riccati',d)[0]
    else:
       [N,beta]=DimRedScenariosNormal(zeros((n_,1)),rho2,k_,j_,'Riccati',d)

    # Step 4. Inverse
    if k_!=0:
       delta2=diag(eye((n_))-beta@beta.T)
       omega2=diag(1/delta2)
       rho2_inv=omega2-omega2@beta/(beta.T@omega2@beta+eye((k_)))@beta.T@omega2
    else:
        rho2_inv=solve(rho2, eye(rho2.shape[0]))

    #Step 5. Cholesky
    rho_inv=cholesky(rho2_inv).T

    #Step 6. Normalizer
    M=sqrt(npsum((rho_inv@N)**2,0))

    #Step 7. Output
    Y=rho_inv@N@diagflat(1/M)
    X=tile(mu, (1,j_))+diagflat(sqrt(diag(sig2)))@N@diagflat(1/M)@diagflat(R)
    return X,R,Y
