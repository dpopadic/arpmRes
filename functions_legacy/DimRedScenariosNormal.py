import matplotlib.pyplot as plt
from numpy import zeros, mod, diag, eye, sqrt, tile, r_, concatenate

plt.style.use('seaborn')

from FactorAnalysis import FactorAnalysis
from NormalScenarios import NormalScenarios


def DimRedScenariosNormal(mu,sig2,k_,j_,method='Riccati',d=None):
    # This function generates Monte Carlo Scenarios from a multivariate normal
    # distribution with mean mu and covariance matrix sig2 through a dimension reduction
    # algorithm resorting to a linear factor model, where the matrix of loadings
    # is recovered through factor analysis of the correlation matrix
    # INPUTS
    # mu        :  [vector] (n_ x 1)  target mean
    # sig2      :  [matrix] (n_ x n_) target positive definite covariance matrix
    # k_        :  [scalar] number of factors to be considered for factor analysis (we reccomend k_ << n_)
    # j_        :  [scalar] Number of scenarios. If not even, j_ <- j_+1
    #   method  :  [string] Riccati (default), CPCA, PCA, LDL-Cholesky, Gram-Schmidt
    #   d       :  [matrix] (k_ x n_) full rank constraints matrix for CPCA
    # OUTPUTS
    # X         :  [matrix] (n_ x j_) panel of MC scenarios drawn from normal
    #                       distribution with mean mu and covariance matrix sig2
    # beta      :  [matrix] (optional) (n_ x k_) loadings matrix ensuing from factor
    # analysis of the correlation matrix
    #
    # For details on the exercise, see here .

    ## Code

    if mod(j_,2)!=0:
        j_=j_+1

    n_=sig2.shape[1]

    #Step 1. Correlation
    rho2=diag(diag(sig2)**(-1/2))@sig2@diag(diag(sig2)**(-1/2))

    #Step 2. Factor Loadings
    _, beta, _, _, _= FactorAnalysis(rho2,zeros((1,1)),k_)

    #Step 3. Residual Standard Deviation
    delta=sqrt(diag(eye((n_))-beta@beta.T))

    #Step 4. Systematic scenarios
    #sigm = r_[-1,r_[eye(k_),zeros((k_,n_))],r_[zeros((n_,k_)),eye(n_)]]
    sigm = concatenate(( concatenate( (eye(k_),zeros((k_,n_)) ) ,axis=1) , concatenate((zeros((n_,k_)),eye(n_)),axis=1) ))
    S,_=NormalScenarios(zeros((k_+n_,1)),sigm,j_,method,d)
    Z_tilde=S[:k_,:]

    #Step 5. Idiosyncratic scenarios
    U_tilde=S[k_:k_+n_,:]

    #Step 6. Output
    X=tile(mu, (1,j_))+diag(sqrt(diag(sig2)))@(beta@Z_tilde+diag(delta)@U_tilde)
    return X, beta

