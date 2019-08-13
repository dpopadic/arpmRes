from collections import namedtuple

from numpy import zeros, tile

from CopMargComb import CopMargComb
from Tscenarios import Tscenarios
from CopMargSep import CopMargSep


def TCopMargScenariosDimRed(nu,rho2,x,u,k_,j_,method='Riccati',d=None):
    # This function generates scenario from a copula-marginal distribution having a Student-t
    # copula and given marginal distributions. Elliptical scenarios used for
    # generating the grades are obtained through dimension reduction.
    #  INPUTS
    #   rho2    : [matrix] (n_ x n_) correlation matrix associated to the t-copula
    #   nu      : [scalar] degrees of freedom associated to the t-copula
    #   x       : [matrix] (n_ x m_) grid of sorted significant nodes, i.e. x(n,j) <=
    #                     x(n,j+1)
    #   u       : [matrix] (n_ x m_) marginal cdf grid: u(n,j)=Fn((x(n,j))
    #   k       : [scalar] number of factors to be used in dimension reduction
    #                    (k=0 means no dimension reduction)
    #   j_      : [scalar] number of simulations
    #   method  : [string] Riccati (default), CPCA, PCA, LDL-Cholesky, Gram-Schmidt
    #   d       : [matrix] (k_ x n_) full rank constraints matrix for CPCA
    #  OPS
    #   X       : [scalar] (n_ x j_) matrix of scenarios coming from a
    #                     copula-marginal distribution with t-copula tCop((rho2,nu)) and given
    #                      marginal distributions

    # For details on the exercise, see here .

    ## Code

    #Step 1. Elliptical Scenarios
    n_,_=rho2.shape
    optionT = namedtuple('options',['dim_red','stoc_rep'])
    optionT.dim_red=k_
    optionT.stoc_rep=0
    X_tilde=Tscenarios(nu,zeros((n_,1)),rho2,j_,optionT,method,d)

    #Step 2. Grades scenarios
    p=tile(1/j_,(1,j_))
    _,_,U=CopMargSep(X_tilde,p)

    #Step 3. Output
    X=CopMargComb(x,u,U)
    return X
