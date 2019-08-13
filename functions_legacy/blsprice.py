# import cython
from scipy.special import erf
from numpy import log, exp, sqrt

# cimport cython
# from libc.math cimport exp, sqrt, pow, log, erf
#
# @cython.cdivision(True)
# cdef double std_norm_cdf(double x) nogil:
#     return 0.5*(1+erf(x/sqrt(2.0)))
#
# @@cython.cdivision(True)
# def blsprice(double s, double k, double t, double v,
#                  double rf, double div, double cp):
#     """Price an option using the Black-Scholes model.
#
#     s : initial stock price
#     k : strike price
#     t : expiration time
#     v : volatility
#     rf : risk-free rate
#     div : dividend
#     cp : +1/-1 for call/put
#     """
#     cdef double d1, d2, optprice
#     with nogil:
#         d1 = (log(s/k)+(rf-div+0.5*pow(v,2))*t)/(v*sqrt(t))
#         d2 = d1 - v*sqrt(t)
#         optprice = cp*s*exp(-div*t)*std_norm_cdf(cp*d1) - \
#             cp*k*exp(-rf*t)*std_norm_cdf(cp*d2)
#     return optprice


def std_norm_cdf(x):
    return 0.5*(1+erf(x/sqrt(2.0)))


def blsprice(s, k, rf, t, v, div=0, cp=1):
    d1 = (log(s / k) + (rf - div + 0.5*pow(v, 2)) * t) / (v * sqrt(t))
    d2 = d1-v*sqrt(t)
    optprice = cp*s*exp(-div*t)*std_norm_cdf(cp*d1) - \
           cp*k*exp(-rf*t)*std_norm_cdf(cp*d2)
    return optprice