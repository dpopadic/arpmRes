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

# # s_bias_vs_variance_lognormal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bias_vs_variance_lognormal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_bias_vs_variance_lognormal).

# +
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bias_vs_variance_lognormal-parameters)

# +
n_samples = 30  # number of samples
mu_x = 0
mu_z = 0
sig2_x = 0.1
sig2_z = 0.5
rho_xz = -0.9
pol_degree = 6  # maximum degree of polynomials considered
j_ = 5000  # simulations of out-of-sample error

mu = np.array([mu_z, mu_x])  # expectation
sig2 = np.array([[sig2_z, np.sqrt(sig2_z)*np.sqrt(sig2_x)*rho_xz],
                 [np.sqrt(sig2_z)*np.sqrt(sig2_x)*rho_xz, sig2_x]])  # cov.

sample = np.exp(np.random.multivariate_normal(mu, sig2, n_samples))

z_in = sample[:, 0]
x_in = sample[:, 1]
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_bias_vs_variance_lognormal-implementation-step00): Initialize variables

# +
err_in = np.zeros(pol_degree)
err_out = np.zeros((j_, pol_degree))
err_out_med = np.zeros(pol_degree)
err_out_iqr = np.zeros(pol_degree)

for deg in np.arange(pol_degree):

    # Step 2: Construct interactions in-sample

    poly = PolynomialFeatures(degree=deg+1, include_bias=False)
    z_inter_in = poly.fit_transform(z_in.reshape(-1, 1))

    # Step 3: Fit conditional expectation through regression in-sample

    reg = linear_model.LinearRegression()
    exp_in_sample = reg.fit(z_inter_in, x_in).predict(z_inter_in)

    # Step 4: Compute in-sample error

    err_in[deg] = np.mean((x_in-exp_in_sample)**2)

    # Step 5: Compute distribution of out-of-sample error

    for j in range(j_):

        # generate out-of-sample
        outsample = np.exp(np.random.multivariate_normal(mu, sig2, n_samples))

        z_out = outsample[:, 0]
        x_out = outsample[:, 1]

        # z_out = np.exp(np.random.normal(mu[0], sig2[0, 0], n_samples))
        poly = PolynomialFeatures(degree=deg+1, include_bias=False)
        z_inter_out = poly.fit_transform(z_out.reshape(-1, 1))

        # error
        exp_out_sample = reg.predict(z_inter_out)
        err_out[j, deg] = np.mean((x_out-exp_out_sample)**2)

    err_out_med[deg] = np.median(err_out[:, deg])
    err_out_iqr[deg] = np.percentile(err_out[:, deg], 75) -\
        np.percentile(err_out[:, deg], 25)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_bias_vs_variance_lognormal-implementation-step06): Conditional expectation

chi = lambda z: np.exp(mu_x +
                       rho_xz*np.sqrt(sig2_x)/np.sqrt(sig2_z) *
                       (np.log(z)-mu_z) + 0.5*(1-rho_xz**2)*sig2_x)

# ## Plots

# +
plt.style.use('arpm')

darkgrey = [.1, .1, .1]
orange = [1, 153/255, 51/255]

xxlim = [0, np.percentile(z_in, 95)*(1+0.5)]
yylim = [0, np.percentile(x_in, 95)*(1+0.5)]

zz = np.arange(10**-3, xxlim[1], 10**-2)
zz_inter = poly.fit_transform(zz.reshape(-1, 1))
xx_hat = reg.fit(z_inter_in, x_in).predict(zz_inter)
xx_hat_best = chi(zz)

fig = plt.figure(figsize=(1280.0/72, 720.0/72), dpi=72)
gs = gridspec.GridSpec(3, 3)
ax_scatter = plt.subplot(gs[1:, :])
ax_inerror = plt.subplot(gs[0, :2])
ax_outerror = ax_inerror.twinx()
ax_hist = plt.subplot(gs[0, -1], sharey=ax_outerror)

# Scatter
ax_scatter.plot(z_in, x_in, '.', markersize=5, color=darkgrey)
ax_scatter.plot(zz, xx_hat, linewidth=2, color='g',
                label='Regr. with %d-th order polynomial' % (deg+1))
ax_scatter.plot(zz, xx_hat_best, color=orange, linewidth=2,
                label='Conditional expectation')
ax_scatter.set_xlim(xxlim)
ax_scatter.set_xlabel('Z', fontsize=14)
ax_scatter.set_ylabel('X', fontsize=14)
ax_scatter.set_ylim(yylim)
ax_scatter.plot(-1, -1, '.', color='k', markersize=0,
                label='$\\rho = %.2f$' % rho_xz)
ax_scatter.legend(loc='upper right', fontsize=14)
ax_scatter.set_title('Joint distribution', fontsize=20, fontweight='bold')

# Errors
# In-sample
ax_inerror.plot(np.arange(1, deg+2), np.log(err_in), color='k',
                label='log-in-sample error')
# Out-of-sample
ax_outerror.plot(np.arange(1, deg+2), np.log(err_out_med), color='r',
                 label='log-out-of-sample error (median)')
ax_outerror.tick_params(axis='y', colors='r')
ax_outerror.grid(False)
ax_outerror.legend(loc='upper right', fontsize=13)
ax_inerror.set_xlabel('Order of polynomial', fontsize=14)
ax_inerror.set_xticks(np.arange(1, deg+2))
ax_inerror.legend(loc='upper left', fontsize=13)
ax_inerror.set_title('Log-errors', fontsize=20, fontweight='bold')

# Histogram
ax_hist.hist(np.log(err_out[:, deg]), bins=int(20*np.log(j_)),
             orientation='horizontal', align='mid', density=True, color='r')
ax_hist.set_title('Log-out-of-sample error', fontsize=20, fontweight='bold')
ax_hist.set_xticks([])
ax_hist.tick_params(axis='y', colors='r')

yylimhist = [np.log(err_out_med).min()-np.abs(np.log(err_out_med).min())/3,
             np.log(err_out_med).max()+np.abs(np.log(err_out_med).max())/3]
ax_hist.set_ylim(yylimhist)

add_logo(fig)
plt.tight_layout()
