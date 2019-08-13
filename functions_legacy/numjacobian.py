import matplotlib.pyplot as plt
from numpy import ones, eye, meshgrid, r_, sort, tile, arange, zeros, min as npmin, max as npmax, sum as npsum, array
from numpy.core.umath import sqrt
from numpy.linalg import qr, solve
import numpy as np

plt.style.use('seaborn')


def numjacobian(fun, x0):
    # numjacobian: estimate of the Jacobian matrix of a vector valued function of n variables
    # usage: jac,err = numjacobian((fun,x0))
    #
    # arguments: (input)
    #  fun - (vector valued) analytical function to differentiate.
    #        fun must be a function of the vector or array x0.
    #
    #  x0  - vector location at which to differentiate fun
    #        If x0 is an nxm array, then fun is assumed to be
    #        a function of n@m variables.
    #
    #
    # arguments: (output)
    #  jac - array of first partial derivatives of fun.
    #        Assuming that x0 is a vector of length p
    #        and fun returns a vector of length n, then
    #        jac will be an array of size (n,p)
    #
    #  err - vector of error estimates corresponding to
    #        each partial derivative in jac.
    #
    # Author: John D.TErrico (woodchips@rochester.rr.com)
    # Version: 03/06/2007
    # Original name: jacobianest

    # get the length of x0 for the size of jac
    nx = x0.size

    MaxStep = 100
    StepRatio = 2.0000001

    # was a string supplied?
    if isinstance(fun, str):
        raise ValueError('fun must be a function, not a string')

    # get fun at the center point
    f0 = fun(x0)
    f0 = f0.flatten()
    n = len(f0)
    if n == 0:
        # empty begets empty
        jac = zeros((1, nx))
        err = jac
        return None

    relativedelta = MaxStep * StepRatio ** arange(0, -25 + -1, -1)
    nsteps = len(relativedelta)

    # total number of derivatives we will need to take
    jac = zeros((n, nx))
    err = jac.copy()
    for i in range(nx):
        x0_i = x0[i]
        if x0_i != 0:
            delta = x0_i * relativedelta
        else:
            delta = relativedelta

        # evaluate at each step, centered around x0_i
        # difference to give a second order estimate
        fdel = zeros((n, nsteps))
        for j in range(nsteps):
            fdif = fun(swapelement(x0.copy(), i, x0_i + delta[j])) - fun(swapelement(x0.copy(), i, x0_i - delta[j]))

            fdel[:, j] = fdif.flatten()

        # these are pure second order estimates of the
        # first derivative, for each trial delta.
        derest = fdel * tile(0.5 / delta, (n, 1))

        # The error term on these estimates has a second order
        # component, but also some 4th and 6th order terms in it.
        # Use Romberg exrapolation to improve the estimates to
        # 6th order, as well as to provide the error estimate.

        # loop here, as rombextrap coupled with the trimming
        # will get complicated otherwise.
        for j in range(n):
            der_romb, errest = rombextrap(StepRatio, derest[j, :], [2, 4])

            # trim off 3 estimates at each  of the scale
            nest = len(der_romb)
            trim = r_[arange(1, 4), nest + arange(-2, 1)]
            der_romb, tags = sort(der_romb), np.argsort(der_romb)
            np.delete(der_romb, trim)
            np.delete(tags, trim)

            errest = errest[tags]

            # now pick the estimate with the lowest predicted error
            err[j, i], ind = npmin(errest), np.argmin(errest)
            jac[j, i] = der_romb[ind]

    return jac, err


# =======================================
#      sub-functions
# =======================================
def swapelement(vec, ind, val):
    # swaps val as element ind, into the vector vec
    vec[ind] = val

    return vec  # sub-function


# ============================================
# subfunction - romberg extrapolation
# ============================================
def rombextrap(StepRatio, der_init, rombexpon):
    # do romberg extrapolation for each estimate
    #
    #  StepRatio - Ratio decrease in step@
    #  der_init - initial derivative estimates
    #  rombexpon - higher order terms to cancel using the romberg step
    #
    #  der_romb - derivative estimates returned
    #  errest - error estimates
    #  amp - noise amplification factor due to the romberg step

    srinv = 1 / StepRatio

    rombexpon = array(rombexpon)

    # do nothing if no romberg terms
    nexpon = len(rombexpon)
    rmat = ones((nexpon + 2, nexpon + 1))
    # two romberg terms
    rmat[1, 1:3] = srinv ** rombexpon
    rmat[2, 1:3] = srinv ** (2 * rombexpon)
    rmat[3, 1:3] = srinv ** (3 * rombexpon)

    # qr factorization used for the extrapolation as well
    # as the uncertainty estimates
    qromb, rromb = qr(rmat)

    # the noise amplification is further amplified by the Romberg step.
    # amp = cond(rromb)

    # this does the extrapolation to a zero step size.
    ne = len(der_init)
    rhs = vec2mat(der_init, nexpon + 2, ne - (nexpon + 2))
    rombcoefs = solve(rromb, qromb.T @ rhs)
    der_romb = rombcoefs[0].T

    # uncertainty estimate of derivative prediction
    s = sqrt(npsum((rhs - rmat @ rombcoefs) ** 2, 0))
    rinv = solve(rromb, eye(nexpon + 1))
    cov1 = npsum(rinv ** 2, 1)  # 1 spare dof
    errest = s.T * 12.7062047361747 ** sqrt(cov1[0])

    return der_romb, errest


# ============================================
# subfunction - vec2mat
# ============================================
def vec2mat(vec, n, m):
    # forms the matrix M, such that M(i,j) = vec(i+j-1)
    i, j = meshgrid(arange(m), arange(n))
    ind = i + j
    mat = vec[ind]
    if n == 1:
        mat = mat.T
    return mat