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

# # s_aggregation_quad [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_aggregation_quad&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-normal-quad-approx).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from arpym.tools import histogram_sp
from arpym.tools.logo import add_logo
from arpym.statistics import saddle_point_quadn
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_quad-parameters)

h = np.array([100000, 80000])  # portfolio holdings
a_pi = -1500  # boundaries of the grid for the pdf
b_pi = 1500

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_quad-implementation-step00): Upload data

# +
path = '../../../databases/temporary-databases'

df = pd.read_csv(path + '/db_pricing_zcb.csv', header=0)
y_tnow = np.array(df['y_tnow'].dropna(axis=0, how='all'))
v_zcb_tnow = np.around(
    np.array(
        df['v_zcb_tnow'].dropna(
            axis=0,
            how='all')),
    decimals=8)
j_, _ = df.shape  # number of scenarios
d_ = len(y_tnow)  # number of key-rates
n_ = len(v_zcb_tnow)  # number of instruments
time2hor_tnow = float(df['time2hor_tnow'].dropna(axis=0, how='all'))  # horizon

# expectation of the risk-drivers at horizon
mu_thor = np.array(df['mu_thor'].dropna(axis=0, how='all'))

# variance of the risk-drivers at horizon
sig2_thor = np.array(df['sig2_thor'].dropna(axis=0, how='all')).reshape(d_, d_)

# transition matrix
theta = np.array(df['theta'].dropna(axis=0,
                 how='all')).reshape(d_, d_)

# scenarios for the ex-ante P&L's
pl = np.array([df['pl' + str(i + 1)] for i in range(n_)]).T

# times to maturity of the instruments
time2mat_tnow = np.array(df['time2mat_tnow'].dropna(axis=0, how='all'))


y_hat = np.array(df['y_hat'].dropna(axis=0, how='all'))


v_zcb_up = np.array(df['bond_t_up'].dropna(axis=0, how='all')).reshape(d_, n_)


v_zcb_down = np.array(df['bond_t_down'].dropna(axis=0,
                      how='all')).reshape(d_, n_)

dur_hat = np.array(df['dur_hat'].dropna(axis=0,
                   how='all')).reshape(d_, n_)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_quad-implementation-step01): Numerical convexities

# +
# terms of the convexity corresponding to parallel shifts
conv_hat = np.zeros((d_, n_))

# numerical differentiation steps
dx = 0.001
dt = 0.001

for d in range(d_):
    # key rates convexities
    conv_hat[d, :] = (v_zcb_up[d, :] -
                      2 * v_zcb_tnow +
                      v_zcb_down[d, :]) / (v_zcb_tnow * dx ** 2)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_quad-implementation-step02): quadratic-normal pdf of the portfolio's P&L distribution

# +
# parameters of the normal-quadratic approx
a_n = v_zcb_tnow * y_hat * time2hor_tnow
a_htilde = a_n@h

b_n = -v_zcb_tnow * dur_hat
b_htilde = b_n@h

c = (v_zcb_tnow * conv_hat / 2)
c_h_tilde = np.diag(h@c.T)

# risk drivers expectation
mu_x_thor = (expm(-theta.dot(time2hor_tnow * 252)) -
             np.eye(d_))@np.squeeze(y_tnow) + mu_thor

# grid of values for the pdf
n_bins = int(round(15 * np.log(1000)))  # number of histogram bins
a_tilde = a_htilde + b_htilde.T@mu_x_thor + mu_x_thor.T@c_h_tilde@mu_x_thor

# portfolio P&L expectation
mu_pi_h = a_tilde + np.trace(c_h_tilde@sig2_thor)
sig2_pi_h = 2 * np.trace((c_h_tilde@sig2_thor)**2) +\
    b_htilde@sig2_thor@b_htilde
grid_pi_h = np.linspace(a_pi, b_pi, n_bins)

# quantiles
quantile_quadn = mu_pi_h + np.sqrt(time2hor_tnow * 252) * grid_pi_h
_, pdf_quadn = saddle_point_quadn(
    quantile_quadn, a_htilde, b_htilde, c_h_tilde, mu_x_thor, sig2_thor)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_quad-implementation-step03): Scenarios for the portfolio P&L and its expectation

pi_h = pl@h
mu_pi_h = np.mean(pi_h)

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure()
lgray = [.8, .8, .8]  # light gray
rescale_pdf = 200000
pdf_mc, bin_mc = histogram_sp(pi_h, p=(1 / j_ * np.ones(j_)), k_=n_bins)

# histogram obtained from exact pricing
plt.barh(bin_mc, pdf_mc * rescale_pdf, left=time2hor_tnow * 252,
         height=bin_mc[1] - bin_mc[0], facecolor=lgray,
         edgecolor=lgray, lw=2)

# saddle point approximation of the Quadn pdf
plot1 = plt.plot(time2hor_tnow * 252 + pdf_quadn * rescale_pdf,
                 quantile_quadn, color='r')

# exact repricing expectation
plot2 = plt.plot(time2hor_tnow * 252, mu_pi_h, color='b', marker='o', lw=2,
                 markersize=5, markeredgecolor='b', markerfacecolor='b',
                 label='Exact repricing')

# saddle point expectation
plot3 = plt.plot(time2hor_tnow * 252, mu_pi_h, color='r', marker='o',
                 lw=2, markersize=3, markeredgecolor='r', markerfacecolor='r',
                 label='Quadratic-normal approx')

plt.xticks(np.arange(0, np.max(time2hor_tnow * 252) + 21, 21))
plt.xlim([0, np.max(time2hor_tnow * 252) + 70])
plt.ylim([min(np.min(bin_mc), np.min(quantile_quadn)),
          max(np.max(bin_mc), np.max(quantile_quadn))])
plt.title('Quadratic-normal approximation ($\\Delta t$=%2.0f days)' %
          (time2hor_tnow * 252))
plt.xlabel('days')
plt.ylabel(r'$\Pi_{h}$')
plt.legend()

add_logo(fig)
plt.tight_layout()
