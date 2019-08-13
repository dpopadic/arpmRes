import numpy as np
from numpy import arange, trace, ones, zeros, real, sign, pi, where, eye, log, exp, sqrt
from numpy import min as npmin
from numpy.linalg import solve, norm, det, pinv
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import norm

np.seterr(all='ignore')


def equation(w, y, c, sigma2, a_tilde, b_tilde, n_):
    # define the equation to compute the saddlepoint
    cprime_Y = a_tilde + trace((c @ sigma2).dot(pinv(eye(n_) - 2 * w * c @ sigma2))) + b_tilde.T @ (
        (w * sigma2 - (w ** 2) * sigma2 @ c @ sigma2).dot(
            pinv((eye(sigma2.shape[0]) - 2 * w * c @ sigma2) @ (eye(sigma2.shape[0]) - 2 * w * c @ sigma2)))) @ b_tilde
    output = cprime_Y - y
    return output.squeeze()


# log-cumulant second derivative function
def cder2(w, c, sigma2, b_tilde, n_):
    cder2_Y = 2 * trace(
        ((c @ sigma2) @ (c @ sigma2)).dot(pinv((eye(n_) - 2 * w * c @ sigma2) @ (eye(n_) - 2 * w * c @ sigma2)))) + \
              (b_tilde.T @ sigma2).dot(pinv((eye(n_) - 2 * w * c @ sigma2) @
                                            (eye(n_) - 2 * w * c @ sigma2) @
                                            (eye(n_) - 2 * w * c @ sigma2))) @ b_tilde
    return cder2_Y.squeeze()


# fix cdf
def check_cdf(cdf, y):
    cdf[0] = npmin(cdf[cdf >= 0])
    index = where((cdf[0] < 0) | (cdf[0] > 1))  # fixes nonsensical values
    cdf_fix = cdf
    mask = ones(cdf.shape, dtype=np.bool)
    mask[0, index] = False
    cdf_fix = cdf_fix[mask]
    idx_y = arange(len(y[0]))
    mask = ones(idx_y.shape, dtype=np.bool)
    mask[index] = False
    idx_y = idx_y[mask]
    # pchip = PchipInterpolator(idx_y, cdf_fix)
    pchip = interp1d(idx_y, cdf_fix)
    cdf_fix = pchip(index)  # preserves monotonicity
    cdf[0, index] = cdf_fix
    return cdf


def SaddlePointQuadN(y, a, b, c, mu, sigma2, w_hat_0=0, threshold=10 ** -1):
    # Saddle-point approximation of the cdf and the pdf of quadratic-normal variable
    #  INPUTS
    #   y         [vector]: (1 x j_) grid of values for y
    #   a         [scalar]: scalar coefficient normal-quadratic approx
    #   b         [vector]: (n_ x 1) vector of coefficient normal-quadratic approx
    #   c         [matrix]: (n_ x n_) matrix of coefficient normal-quadratic approx
    #   mu        [vector]: (n_ x 1) expectations of normal variables
    #   sigma2    [matrix]: (n_ x n_) covariances of normal variables
    #   w_hat_0   [scalar]: (optional) starting point for optimization
    #   threshold [scalar]: (optional) threshold for interpolation of u and r when are near 0
    #  OPS
    #   F_y       [vector]: (1 x j_) Saddle-point cdf
    #   f_y       [vector]: (1 x j_) Saddle-point pdf
    # Due to numerical issues, you could experience some
    # irregularities in the computed cdf. In this case, try to set higher values
    # for the threshold (which is 10**-1 as default).

    # For details on the exercise, see here .

    ## Code

    n_ = len(mu)  # number of variables
    leny = y.shape[1]

    # transformation of the parameters
    a_tilde = a + b.T @ mu + mu.T @ c @ mu
    b_tilde = b + 2 * c @ mu
    w_hat = zeros((1, leny))
    cder1_Y = zeros((1, leny))

    test1_cder2 = zeros((1, leny))
    test2_cder2 = zeros((1, leny))

    # find the roots (zeros) of the equation cder1_Y(w_hat) - y = 2
    for i in range(leny):
        if n_ == 1:  # solve the quadratic equation analitically
            a_ = -(b_tilde ** 2) @ sigma2 @ c @ sigma2 + 4 * (a_tilde - y[0, i]) @ (c ** 2) @ (sigma2 ** 2)
            b_ = -2 * (c ** 2) @ (sigma2 ** 2) + (b_tilde ** 2) @ sigma2 - 4 * (a_tilde - y[0, i]) @ c @ sigma2
            c_ = c @ sigma2 + (a_tilde - y[0, i])
            delta_ = b_ ** 2 - 4 * a_ @ c_

            if delta_ < 0:
                delta_ = 0

            w_hat1 = (-b_ + sqrt(delta_)) / (2 * a_)
            w_hat2 = (-b_ - sqrt(delta_)) / (2 * a_)
            test1_cder2[0, i] = cder2(w_hat1, c, sigma2, b_tilde, n_)
            test2_cder2[0, i] = cder2(w_hat2, c, sigma2, b_tilde, n_)

            if test1_cder2[0, i] >= 0 and test2_cder2[0, i] < 0:
                w_hat[0, i] = w_hat1
            elif test2_cder2[0, i] >= 0 and test1_cder2[0, i] < 0:
                w_hat[0, i] = w_hat2
            elif test1_cder2[0, i] >= 0 and test2_cder2[0, i] >= test1_cder2[0, i]:
                w_hat[0, i] = w_hat2
            elif test2_cder2[0, i] >= 0 and test1_cder2[0, i] >= test2_cder2[0, i]:
                w_hat[0, i] = w_hat1
            else:
                w_hat[0, i] = np.NaN
            cder1_Y[0, i] = equation(w_hat[0, i], y[0, i], c, sigma2, a_tilde, b_tilde, n_)
        else:  # solve numerically the equation cder1_Y(w_hat) - y = 0
            test_cder2 = cder2(w_hat_0, c, sigma2, b_tilde, n_)

            if test_cder2 >= 0:
                # options = optimset(Display, off)
                x, info, _, _ = fsolve(lambda w_hat: equation(w_hat, y[0, i], c, sigma2, a_tilde, b_tilde, n_), w_hat_0,
                                       full_output=True)
                w_hat[0, i], cder1_Y[0, i] = x, info['fvec']
            else:
                w_hat_0 = -w_hat_0
                # options = optimset(Display, off)
                x, info, _, _ = fsolve(lambda w_hat: equation(w_hat, y[0, i], c, sigma2, a_tilde, b_tilde, n_), w_hat_0,
                                       full_output=True)
                w_hat[0, i], cder1_Y[0, i] = x, info['fvec']

            w_hat_0 = w_hat[0, i]
    # delete a point
    idxnan = sum(np.isnan(w_hat))
    w_hat[0, idxnan] = np.NaN

    # log-cumulant-generating function and its second derivative
    c_Y = zeros((1, leny))
    cder2_Y = zeros((1, leny))
    for i in range(leny):
        if np.isnan(w_hat[0, i]):
            c_Y[0, i] = np.NaN
            cder2_Y[0, i] = np.NaN
        else:
            aux = (eye(n_) - 2 * w_hat[0, i] * sigma2 @ c)
            aux = solve(aux, sigma2)
            c_Y[0, i] = w_hat[0, i] * a_tilde - 0.5 * log(det(eye(n_) - 2 * w_hat[0, i] * sigma2 @ c)) + 0.5 * (
                        w_hat[0, i] ** 2) * b_tilde.T @ aux @ b_tilde
            cder2_Y[0, i] = cder2(w_hat[0, i], c, sigma2, b_tilde, n_)

    c_Y = real(c_Y)
    r = sign(w_hat) * sqrt(2) * sqrt(w_hat * y - c_Y)
    r = real(r)
    check_cder2 = where(cder2_Y < 0)
    cder2_Y[check_cder2] = np.NaN
    u = w_hat * sqrt(cder2_Y)
    d = (1 / u) - (1 / r)

    # linear interpolation of d when r and u are close to 0
    d_ = d
    i_r = arange(leny)
    index_r = where((r > -threshold) & (r < threshold))[1]
    dd_u = d
    i_u = arange(leny)
    index_u = where((u > -threshold) & (u < threshold))[1]
    if len(index_r) > leny / 2:
        print('Check the threshold for r')
        F_y = zeros((leny, 1))
    elif len(index_u) > leny / 2:
        print('Check the threshold for u')
        F_y = zeros((leny, 1))
    else:
        mask = np.ones(d_.shape, dtype=bool)
        mask[0, index_r] = False
        d_ = d_[mask]
        mask = np.ones(i_r.shape, dtype=bool)
        mask[index_r] = False
        i_r = i_r[mask]
        interp = interp1d(i_r, d_, fill_value='extrapolation')
        di = interp(index_r)
        d[0, index_r] = di
        mask = np.ones(dd_u.shape, dtype=bool)
        mask[0, index_u] = False
        dd_u = dd_u[mask]
        mask = np.ones(i_u.shape, dtype=bool)
        mask[index_u] = False
        i_u = i_u[mask]
        interp = interp1d(i_u, dd_u, fill_value='extrapolation')
        di_u = interp(index_u)
        d[0, index_u] = di_u
        F_y = norm.cdf(r) - norm.pdf(r) * d
        F_y = check_cdf(F_y, y)

    # Compute density
    f_y = (2 * pi) ** -0.5 * (cder2_Y) ** -0.5 * exp(c_Y - w_hat * y)
    return F_y, f_y
