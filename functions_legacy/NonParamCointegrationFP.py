from numpy import arange, sqrt, r_
from numpy import sum as npsum
from numpy.linalg import solve

from pcacov import pcacov
from FitVAR1 import FitVAR1
from VAR1toMVOU import VAR1toMVOU
from FPmeancov import FPmeancov


def NonParamCointegrationFP(x, p,time_step, theta_threshold):
    ## This function estimates the cointegrated vectors of a multivariate process
    #  INPUTS
    #   x                :[matrix](d_ x t_end) historical series
    #   p                :[vector](1 x t_end) historical Flexible Probabilities
    #   time_step        :[scalar] estimation step
    #   theta_threshold  :[scalar] positive threshold for stationarity test
    #  OPS
    #   c                :[matrix](d_ x k_) k_ cointegrated eigenvectors
    #   y_hat            :[matrix](k_ x t_end) cointegrated time series
    #   lam_y            :[vector](k_ x 1) eigenvectors corresponding to cointegrated eigenvectors
    #   mu_hat           :[vector](k_ x 1) estimated long-run expectation
    #   theta            :[matrix](k_ x 1) estimated transition parameter
    #   sd_hat           :[vector](k_ x 1) estimated long-run standard deviation

    # For details on the exercise, see here .
    ## Code

    # number of variables
    d_ = x.shape[0]

    # estimate HFP covariance matrix
    _,sigma2_hat = FPmeancov(x,p)

    # pca decomposition
    e_hat, lam_hat = pcacov(sigma2_hat)

    # define series
    y_t = e_hat.T@x

    # fit the series
    k=0
    for d in arange(d_-1,-1,-1):
        # fit the series with an univariate OU process
        alpha, b, sig2_U = FitVAR1(y_t[[d],:],p[[0],:-1]/npsum(p[0,:-1]))
        mu, theta, sigma2, _=VAR1toMVOU(alpha, b, sig2_U, time_step)
        #[mu, theta, sigma2] = FitVAR1MVOU(dy_t, y_t(d, 1:-1), time_step, p([:-1]/sum(p[:-1])))

        # cointegrated vectors
        if theta> theta_threshold:
            if k==0:
                c = e_hat[:, [d]]
                y_hat = y_t[[d],:]
                lam_y = lam_hat[d]
                mu_hat = solve(theta,mu)
                theta_hat = theta
                sd_hat = sqrt(sigma2/(2*theta))
            else:
                c = r_['-1',c,e_hat[:, [d]]]
                y_hat = r_[y_hat, y_t[[d],:]]
                lam_y = r_[lam_y, lam_hat[d]]
                mu_hat = r_[mu_hat, solve(theta,mu)]
                theta_hat = r_[theta_hat,theta]
                sd_hat = r_[sd_hat,sqrt(sigma2/(2*theta))]
            k = k+1

    return c, y_hat, lam_y, mu_hat, theta_hat, sd_hat
