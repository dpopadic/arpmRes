from numpy import zeros, tile, r_

from scipy.linalg import expm

from ProjMomentsVAR1MVOU import ProjMomentsVAR1MVOU
from NormalScenarios import NormalScenarios


def SimVAR1MVOU(x_0, u, theta, mu, sigma2, j_):
    # Simulate the MVOU process to future horizons by Monte Carlo method
    # model: dXt=-(theta*Xt-mu)dt+sigma*dWt
    #  INPUTS
    #   X_t        [matrix]: (n_ x j_) initial conditions at time t
    #   u          [vector]: (1 x u_)  projection horizons
    #   theta      [matrix]: (n_ x n_) transition matrix
    #   mu         [vector]: (n_ x 1) long-term means
    #   sigma2     [matrix]: (n_ x n_) covariances
    #   j_         [scalar]: simulations number
    #  OPS
    #   X_u        [tensor]: (n_ x j_ x u_) simulated process at times u_

    ## Code

    n_, _ = x_0.shape
    t_=u.shape[1]

    if t_>1:
        tau = r_['-1', u[0,0], u[0,1:]-u[0,:-1]]
    else:
        tau = u.copy()

    X_u = zeros((n_, j_, t_))

    for t in range(t_):
        # project moments from t to t+tau
        mu_tau, sigma2_tau, _ = ProjMomentsVAR1MVOU(zeros((n_,1)), tau[t], mu, theta, sigma2)

        # simulate invariants
        Epsi,_ = NormalScenarios(zeros((n_,1)), sigma2_tau, j_, 'Riccati')

        # simulte MVOU process to future horizon
        if t_>1 and t>1:
            x_0 = X_u[:, :, t-1]
        X_u[:,:,t] = expm(-theta*tau[t])@x_0 + tile(mu_tau, (1, j_)) + Epsi
    return X_u.squeeze()
