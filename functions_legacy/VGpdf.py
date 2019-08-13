from numpy import pi, abs, log, exp

from scipy.special import gamma, kv


def VGpdf(x,par,tau):
    # This function computes the Variance-Gamma pdf at horizon tau, evaluated
    # at points x
    #  INPUTS
    # x    :[vector] points at which the pdf is evaluated
    # par  :[struct] parameters of the VG model the struct has fields {c, m, g}
    # tau  :[scalar] time horizon
    #  OPS
    # y    :[vector] VG pdf

    ## Code

    c = par.c
    m = par.m
    g = par.g

    alpha = c*tau*log(g*m)-2*log(gamma(c*tau))-(2*c*tau-1)*log(g+m)
    b = (g+m)*abs(x)
    beta = log(gamma(c*tau))-log(pi)/2+(c*tau-1/2)*log(b)+b/2+log(kv(c*tau-1/2,b/2))
    ln_pdf = alpha + ((g-m)/2)*x - ((g+m)/2)*abs(x) + beta
    y = exp(ln_pdf)
    return y
