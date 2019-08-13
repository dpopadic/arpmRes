from numpy import arange, unique, exp, sqrt
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.stats import norm, lognorm


def QuantileMixture(p, alpha, mu_Y, sigma_Y, mu_Z, sigma_Z):
    # This function computes the quantiles, corresponding to probability levels
    # p, from a mixture distribution consisting of a linear combination of a
    # normal and a lognormal random variables: f = alpha@f_Y + (1-alpha)f_Z
    # The computation of the quantile uses a linear interpolation if the
    # confidence levels p are uniformly distributed on [0,1], then the sample q
    # is distributed as the mixture:
    # INPUTS
    #  p        :[vector] in [0,1], probability
    #  alpha    :[scalar] in (0,1), mixing coefficient
    #  mu_Y     :[scalar] mean of normal component
    #  sigma_Y  :[scalar] standard deviation of normal component
    #  mu_Z     :[scalar] first parameters of the log-normal component
    #  sigma_Z  :[scalar] second parameter of the log-normal component
    # OUTPUTS
    #  q        :[scalar] quantile

    # compute mean
    mu = alpha*mu_Y + (1 - alpha)*exp(mu_Z + 0.5 * sigma_Z**2)# mean of the mixture distribution

    # compute standard deviation
    s2_Y = mu_Y**2 + sigma_Y**2# second non-central moment of Y
    s2_Z = exp(2*mu_Z + 2*sigma_Z**2)# second non-central moment of Z
    s2 = alpha*s2_Y + (1 - alpha)*s2_Z# second non-central moment of the mixture distribution
    sigma = sqrt(s2 - mu**2)# standard deviation of the mixture distribution

    # compute cdf on suitable range
    x = mu + 6 *sigma*arange(-1,1.001,0.001)
    F = alpha*norm.cdf(x, mu_Y, sigma_Y) + (1 - alpha)*lognorm.cdf(x, sigma_Z, scale=exp(mu_Z))
    F, ind = unique(F, return_index=True)
    x = x[ind]

    # compute quantile by interpolation
    interp = interp1d(F.flatten(), x.flatten(), fill_value='extrapolate')
    q = interp(p)

    return q
