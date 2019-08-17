#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_elltest_ytm_monthly [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_elltest_ytm_monthly&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMRzerorates).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.estimation import fit_var1, fit_lfm_ols
from arpym.statistics import invariance_test_ellipsoid
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_monthly-parameters)

# +
t_ = 1000  # length of time series of yields
tau = 10  # selected time to maturity (years)
l_ = 25  # lag for the ellipsoid test
conf_lev = 0.95  # confidence level for the ellipsoid test
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_monthly-implementation-step00): Load data

# +
tau = np.array([tau])
path = '../../../databases/global-databases/fixed-income/db_yields'
y = pd.read_csv(path + '/data.csv', header=0, index_col=0)
y = y[tau.astype(float).astype(str)]
x = y.values  # yield to maturity
x = x[::20, :]
x = x[-t_:, :].reshape(-1)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_monthly-implementation-step01): AR(1) fit

# +
p = np.ones(x.shape) / x.shape
b_hat, _, _ = fit_var1(x)

# realized invariant
epsi = x[1:] - x[:-1] * b_hat
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_monthly-implementation-step02): ellipsoid tests

# +
plt.style.use('arpm')

# perform and show ellipsoid test for invariance on monthly yield
name1 = 'Invariance test on monthly yield'
acf_x, conf_int_x = \
    invariance_test_ellipsoid(x, l_, conf_lev=conf_lev, fit=0, r=2,
                              title=name1)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)

plt.style.use('arpm')
# perform and show ellipsoid test for invariance on AR(1) residuals
name2 = 'Invariance test on AR(1) residuals'
acf_epsi, conf_int_epsi = \
    invariance_test_ellipsoid(epsi, l_, conf_lev=conf_lev, fit=0,
                              r=2, title=name2)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)

plt.style.use('arpm')
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_monthly-implementation-step03): linear fit on on log-autocorrelations

# +
l_points = np.max([l_, 3])
lag = 1 + np.arange(l_points)
log_acf_x = np.log(acf_x)
# log-autocorrelations linear fit
print(log_acf_x, lag, log_acf_x.shape, lag.shape)
alpha, beta, _, _ = fit_lfm_ols(log_acf_x, lag)  # linear fit
# decay coefficient
lambda_hat = -beta

# Linear fit of log-autocorrelation plot
log_acf_fit = beta * np.arange(1, l_points + 0.01, 0.01) + alpha
fig = plt.figure()
pp2 = plt.plot(lag, log_acf_x, color=[.9, .4, 0], lw=1.5)
pp1 = plt.plot(np.arange(1, l_points + 0.01, 0.01), log_acf_fit, lw=1.4)
plt.axis([0, l_points,  np.min(log_acf_x), np.max(log_acf_x)])
plt.gca().yaxis.tick_right()
plt.xlabel('Lag')
plt.ylabel('log-autocorrelation')
plt.xticks()
plt.yticks()
plt.legend(['empirical', ' linear fit\n $\lambda$ = %1.3f' % lambda_hat])
add_logo(fig, location=3)
