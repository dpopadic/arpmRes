from numpy import log, exp, sqrt, real


def HestonChFun(omega, mu, kappa, s2_, eta, rho, x, y, tau):
    ## Compute the characteristic function for the Heston model
    # dX_t = mudt + sqrt(Y_t)dW1_t
    # dY_t = -kappa@(Y_t - s2_)@dt + eta@sqrt(Y_t)@dW2_t, E{X_t Y_t} = rho

    h = 1j * omega * eta * rho - kappa
    r = sqrt(h ** 2 + omega ** 2 * eta ** 2)
    a = -kappa * s2_ / (eta ** 2) * (2 * log((2 * r - (r + h) * (1 - exp(-tau * r))) / (2 * r)) + (r + h) * tau)
    b = (-(omega ** 2) * (1 - exp(-tau * r))) / (r - h + exp(-r * tau) * (r + h))
    c = 1j * omega

    fun = real(exp(c * mu * tau + c * x + a + b * y))
    return fun

