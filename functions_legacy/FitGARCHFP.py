import numpy as np
from numpy import array, ones, zeros, log, sqrt
from numpy import sum as npsum
from scipy.optimize import minimize


def coneq(x, A, b):
    return A.dot(x) - b


def conineq(x, A, b):
    '''
    Ax<=b -> g(x)>0, g(x) = -Ax+b
    '''
    res = (-A.dot(x)).squeeze() + b
    return res


def LL(param, dx, t_, s2_0, p):
    c = param[0]
    a = param[1]
    b = param[2]
    mu = param[3]

    s2 = zeros(t_)
    standardized_dx = zeros(t_)

    s2[0] = c + (a + b) * s2_0
    standardized_dx[0] = ((dx[0, 0] - mu) ** 2) / s2[0]

    for t in range(1, t_):
        s2[t] = c + b * s2[t - 1] + a * dx[0, t - 1] ** 2
        standardized_dx[t] = ((dx[0, t] - mu) ** 2) / s2[t]
    L = npsum(p * log(s2)) + npsum(p * standardized_dx)  # this is -L
    return L


def constr(par, C):
    x = C - npsum(par[1:3])
    return x


def FitGARCHFP(dx, s2_0, p0, g=0.95, p=None):
    # Fit GARCH(1,1) model
    # INPUT
    # dx   :[vector] (1 x t_end) realized increments of the GARCH process X
    # s2_0 :[scalar] initial value for sigma**2
    # p0   :[vector] starting guess for the vector of parameters
    #                 par0[0]: guess for c
    #                 par0[1]: guess for a
    #                 par0[2]: guess for b
    #                 par0[3]: guess for mu
    # g    :[vector] (default: [scalar] 0.95) the constraint a+b <=g is imposed (stationarity condition a+b <1)
    # p:  :[vector] (1 x t_end) flexible probabilities default: the observations are equally weighted

    # OP
    # par  :[matrix] (4 x length([g])) for each constraint g[i], par(0:3, g[i])
    #                gives the estimates of the three parameters c,a,b,mu respectively
    # sig2 :[matrix] (length([g])) x t_end) estimated path of the squared scatter
    # epsi :[matrix] (length([g])) x t_end) residuals
    # lik  :[vector] (length([g])) x 1) maximum likelihood achieved in correspondence of each value of g

    t_ = dx.shape[1]

    if p is None:
        p = 1 / t_ * ones((1, t_))

    if isinstance(g, float):
        leng = 1
        g = array([g])
    else:
        leng = max(g.shape)

    if p0 is None:
        p0 = [0, 0.01, 0.8, 0]

    if len(p0) < 4:
        p0.append(0)

    par = zeros((4, leng))
    lik = zeros((leng))
    sig2 = zeros((leng, dx.shape[1]))
    epsi = zeros((leng, dx.shape[1]))

    ## Repeat MLFP GARCH estimation for each constraint a+b<(=)g
    # [In the default case MLFP GARCH is applied only once]
    for i in range(leng):
        param, fval, _, _, newsig2, newepsi = MLGarch(dx, s2_0, p0, g[i], p, t_)
        par[:, i] = param.T
        sig2[i, :] = newsig2.copy()
        epsi[i, :] = newepsi.copy()
        lik[i] = -fval
    return par, sig2, epsi, lik


## MLFP GARCH estimation function
def MLGarch(dx, s2_0, p0, g, p, t_):
    # constraints a,b,c >(=)0, a+b<(=)g
    A = array([[0, 1, 1, 0]])
    C = g
    options = {'maxiter': 4 * 500, 'ftol': 1e-12, 'disp': False}
    cons = ({'type': 'ineq', 'fun': lambda x, A, C: conineq(x, A, C), 'args': (A, C)})
    res = minimize(LL, p0, args=(dx, t_, s2_0, p), constraints=cons,
                   bounds=((1e-15, 100), (1e-15, 1), (1e-15, 1), (1e-15, 1e15)), options=options)
    if res.status in [4, 8]:
        options = {'maxiter': 4 * 2000, 'ftol': 1e-10, 'disp': False}
        res = minimize(LL, [0, 0.2, 0.7, 1e-5], args=(dx, t_, s2_0, p), constraints=cons,
                       bounds=((1e-15, 100), (1e-15, 1), (1e-15, 1), (1e-15, 1e15)), options=options)
        print(res.message)
    par = res.x

    sig2 = zeros((1, t_))
    eps = zeros((1, t_))

    sig2[0, 0] = s2_0
    eps[0, 0] = dx[0, 0] / sqrt(sig2[0, 0])

    for t in range(1, t_):
        sig2[0, t] = par[0] + par[2] * sig2[0, t - 1] + par[1] * dx[0, t - 1] ** 2
        eps[0, t] = (dx[0, t] - par[3]) / sqrt(sig2[0, t])
    return par, res.fun, res.status, res, sig2, eps


