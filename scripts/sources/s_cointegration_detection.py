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

# # s_cointegration_detection [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cointegration_detection&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_detection).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.estimation import cointegration_fp, fit_var1
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_detection-parameters)

t_in = 1260  # length of the in-sample time series (days)
t_ = 2268  # length of the complete series (in and out-of-sample) (days)
u = 0.35  # coefficient of linear combination
l_select = 3  # selected eigenvector

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_detection-implementation-step00): Load data

# +
tau = np.array([1, 2, 3, 5, 7, 10])
path = '../../../databases/global-databases/fixed-income/db_yields'
x = pd.read_csv(path + '/data.csv', header=0, index_col=0)
x = x[tau.astype(float).astype(str)].tail(t_).values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_detection-implementation-step01): Select the in-sample and out-of-sample series

# +
x_in = x[:t_in, :]  # in-sample series
x_out = x[t_in:, :]  # out-of-sample series
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_detection-implementation-step02): Cointegrated eigenvectors

# +
c_hat, _ = cointegration_fp(x_in)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_detection-implementation-step03): In sample and out-of-sample cointegrated series

# +
# store cointegrated vectors
c_hat_sel = np.zeros((c_hat.shape[0], 3))
c_hat_sel[:, 0] = c_hat[:, l_select+1]
c_hat_sel[:, 1] = c_hat[:, l_select]
c_hat_sel[:, 2] = (1 - u) * c_hat[:, l_select + 1] + u * \
    c_hat[:, l_select]

# in-sample cointegrated series (basis points)
y_in = x_in @ c_hat_sel * 10000
# out-of-sample cointegrated series (basis points)
y_out = x_out @ c_hat_sel * 10000
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_cointegration_detection-implementation-step04): AR(1) long term parameters

# +
exp_infty = np.zeros(3)
sd_infty = np.zeros(3)
tau_halflife = np.zeros(3)

for k in range(3):

    # AR1 fit
    b_hat, mu_hat_epsi, sig2_hat_epsi = fit_var1(y_in[:, [k]])

    # long-run expectation
    exp_infty[k] = mu_hat_epsi / (1 - b_hat)

    # long-run standard deviation
    sd_infty[k] = np.sqrt(sig2_hat_epsi / (1 - b_hat ** 2))

    # half life
    tau_halflife[k] = -np.log(2) / np.log(abs(b_hat))
# -

# ## Plots

# +
plt.style.use('arpm')

for k in range(3):
    fig = plt.figure()
    min_y = min(min(y_in[:, k]), min(y_out[:, k]))
    max_y = max(max(y_in[:, k]), max(y_out[:, k]))

    t = np.arange(t_)/252
    plt.axis([0, t[-1], min_y, max_y])
    plt.xlabel('time (years)')
    plt.ylabel('basis points')
    plt.xticks()
    plt.yticks()
    insample = plt.plot(t[:t_in], y_in[:, k], color='k', linewidth=1)
    outofsample = plt.plot(t[t_in:], y_out[:, k], color='b', linewidth=1)
    expect = plt.plot(t, np.tile(exp_infty[k], t_), color='g')
    up_sd = plt.plot(t, np.tile(exp_infty[k] + 2 * sd_infty[k], t_),
                     color='r')
    plt.plot(t, np.tile(exp_infty[k] - 2 * sd_infty[k], t_),
             color='r')
    plt.legend(handles=[insample[0], expect[0], up_sd[0], outofsample[0]],
               labels=['In-Sample', 'In-Sample Mean',
                       '+/- 2 In-Sample St. Dev', 'Out-of-Sample'], loc=2)

    if k == 0:
        plt.title(('Series = {index}-th Eigvect. In-Sample Mean-Reversion ' +
                   'Half-Life = ' +
                   ' {halflife:.0f} days.').format(index=l_select,
                                                   halflife=tau_halflife[k]))
    elif k == 1:
        plt.title(('Series = {index}-th Eigvect. In-Sample Mean-Reversion ' +
                   'Half-Life = ' +
                   ' {halflife:.0f} days.').format(index=l_select+1,
                                                   halflife=tau_halflife[k]))
    else:
        plt.title(('Series = {a:1.2f} x {index}-th Eigvect. + ' +
                   '{a2:1.2f} x {index2}-th Eigvect.' +
                   '\nIn-Sample Mean-Reversion Half-Life ' +
                   '= {halflife:.0f} days.').format(a=np.sqrt(1-u**2),
                                                    index=l_select,
                                                    a2=u**2,
                                                    index2=l_select+1,
                                                    halflife=tau_halflife[k]))
    add_logo(fig)
    plt.tight_layout()
