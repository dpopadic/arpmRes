import os
import os.path as path
import sys

from numpy import ones, zeros, r_

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from autocorrelation import autocorrelation
from InvarianceTestEllipsoid import InvarianceTestEllipsoid
from InvarianceTestCopula import InvarianceTestCopula
from TestKolSmirn import TestKolSmirn
from InvarianceTestKolSmirn import InvarianceTestKolSmirn
from SWDepMeasure import SWDepMeasure


def InvarianceTests(Epsi, lag_):
    # this function performs the invariance tests on a time series Epsi on the
    # lags lag_
    # Epsi    [1xt_] vector of observations
    # lag_    [scalar] perform the tests on the lags up to lag_

    Epsi = Epsi[[0]]
    t_ = Epsi.shape[1]

    ## Compute the autocorrelations

    acf_Epsi = autocorrelation(Epsi, lag_)

    ## Perform Kolmogorov-Smirnov test

    Epsi_1, Epsi_2, band_int, F_1, F_2, up_band, low_band = TestKolSmirn(Epsi)

    ## Estimate the SW measures of dependence

    dep_Epsi = zeros((lag_, 1))
    for l in range(lag_):
        probs = ones((1, t_ - (l + 1))) / (t_ - (l + 1))
        dep_Epsi[l] = SWDepMeasure(r_[Epsi[[0], (l + 1):], Epsi[[0], : - (l + 1)]], probs)

    ## Plot ellipsoid and autocorrelation coefficients for invariance test

    figure(figsize=(12, 6))
    ell_scale = 2  # ellipsoid radius coefficient
    fit = 0  # normal fitting
    InvarianceTestEllipsoid(Epsi, acf_Epsi[0, 1:], lag_, fit, ell_scale)

    ## Plot Kolmogorov-Smirnov test for invariance

    figure()
    InvarianceTestKolSmirn(Epsi, Epsi_1, Epsi_2, band_int, F_1, F_2, up_band, low_band)

    ## Plot copula pdf and measures of dependence for invariance test

    figure()
    InvarianceTestCopula(Epsi, dep_Epsi, lag_)
