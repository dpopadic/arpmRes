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

# # s_info_processing_comparison [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_info_processing_comparison&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sablepcomparison).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from arpym.estimation import exp_decay_fp
from arpym.statistics import meancov_sp
from arpym.tools import plot_ellipse, add_logo
from arpym.views import black_litterman, min_rel_entropy_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-parameters)

c = 0.82  # confidence level in the views
eta = np.array([1, -1])  # parameters for qualitative views
lam = 1.2  # average risk-aversion level
tau = 252  # uncertainty level in the reference model
tau_hl = 1386  # half life parameter
v = np.array([[1, - 1, 0], [0, 0, 1]])  # pick matrix
w = np.array([1/3, 1/3, 1/3])  # market-weighted portfolio

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-implementation-step00): Upload data

# +
path = '../../../databases/global-databases/equities/db_stocks_SP500/'

data = pd.read_csv(path + 'db_stocks_sp.csv', index_col=0, header=[0, 1])
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-implementation-step01): Compute time series of returns

n_ = len(w)  # market dimension
r_t = data.pct_change().iloc[1:, :n_].values  # returns

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-implementation-step02): Compute the sample mean and the exponential decay sample covariance

t_ = len(r_t)
p_t_tau_hl = exp_decay_fp(t_, tau_hl)  # exponential decay probabilities
mu_hat_r, sig2_hat_r = meancov_sp(r_t, p_t_tau_hl)  # sample mean and cov.

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-implementation-step03): Compute Black-Litterman prior parameters

# +
# expectation in terms of market equilibrium
mu_r_equil = 2 * lam * sig2_hat_r @ w

mu_m_pril = mu_r_equil
cv_pri_pred = (1 + 1 / tau) * sig2_hat_r
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-implementation-step04): Compute Black-Litterman posterior parameters

# +
# vectors quantifying the views
i = v @ mu_r_equil + eta * np.sqrt(np.diag(v @ sig2_hat_r@ v.T))
sig2_i_mu = ((1 - c) / (tau * c)) * (v @ sig2_hat_r @ v.T)

# Black-Litterman posterior parameters
mu_m_pos, cv_pos_pred = black_litterman(mu_r_equil, sig2_hat_r, tau, v, i,
                                        sig2_i_mu)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-implementation-step05): Compute Black-Litterman sure posterior parameters

mu_r_sure_bl = mu_r_equil + sig2_hat_r @ v.T @ \
             np.linalg.solve(v @ sig2_hat_r @ v.T, i - v @ mu_r_equil)
sig2_r_sure_bl = (1 + 1 / tau) * sig2_hat_r - (1 / tau) * sig2_hat_r @ v.T\
               @ np.linalg.solve(v @ sig2_hat_r @ v.T, v @ sig2_hat_r)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-implementation-step06): Compute posterior parameters from distributional views (Minimum Relative Entropy)

# +
v_mre = v
v_sig_mre = np.eye(n_)
imre = i
sig2_i_mumre = sig2_hat_r

mu_r_mre, sig2_r_mre = min_rel_entropy_normal(mu_r_equil, sig2_hat_r, v_mre,
                                              imre, v_sig_mre, sig2_i_mumre)
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_info_processing_comparison-implementation-step07): Compute posterior parameters from point views

# +
k_ = len(v)  # view variables dimension
v_point = v
z_point = i

mu_r_point, sig2_r_point = min_rel_entropy_normal(mu_r_equil, sig2_hat_r,
                                                  v_point, z_point, v_point,
                                                  np.zeros((k_)))
# -

# ## Plots

# +
col_darkgrey = [.6, .6, .6]
col_orange = [.9, .6, 0]
col_reddishpurple = [.8, .6, .7]
col_skyblue = [.35, .7, .9]
spot = [0, 1]

plt.style.use('arpm')
fig = plt.figure(figsize=(10, 8))

plot_ellipse(mu_hat_r[spot], sig2_hat_r[np.ix_(spot, spot)],
             color=col_darkgrey, line_width=1.3)
plot_ellipse(mu_m_pril[spot], cv_pri_pred[np.ix_(spot, spot)], color='k',
             line_width=1.3)
plot_ellipse(mu_m_pos[spot], cv_pos_pred[np.ix_(spot, spot)], color='b',
             line_width=1.3)
plot_ellipse(mu_r_sure_bl[spot], sig2_r_sure_bl[np.ix_(spot, spot)],
             color=col_skyblue, line_width=2)
plot_ellipse(mu_r_mre[spot], 0.98 * sig2_r_mre[np.ix_(spot, spot)],
             color=col_orange, line_width=1.5)
plot_ellipse(mu_r_point[spot], sig2_r_point[np.ix_(spot, spot)],
             color=col_reddishpurple, line_width=1.3)

plt.plot(mu_hat_r[spot[0]], sig2_hat_r[0, spot[1]], '.', color=col_darkgrey,
         markersize=20)
plt.plot(mu_m_pril[spot[0]], mu_m_pril[spot[1]], '*', color='k', markersize=15)
plt.annotate('equilibrium', weight="bold",
             xy=(mu_m_pril[spot[0]], mu_m_pril[spot[1]]),
             xytext=(0.006, 0.01),
             arrowprops=dict(facecolor="black", width=0.5,
                             headwidth=4, shrink=0.1))
plt.plot(mu_m_pos[spot[0]], mu_m_pos[spot[1]], '.', color='b', markersize=15)
plt.plot(mu_r_sure_bl[spot[0]], mu_r_sure_bl[spot[1]], 'o', color=col_orange,
         markersize=10)
plt.plot(mu_r_mre[spot[0]], mu_r_mre[spot[1]], '.', color=col_skyblue,
         markersize=15)
plt.plot(mu_r_point[spot[0]], mu_r_point[spot[1]], '.',
         color=col_reddishpurple, markersize=5)

plt.plot(r_t.T[0], r_t.T[1], '.', color=col_darkgrey, markersize=4)
plt.xticks(np.arange(-0.04,  0.071, step=0.01))

plt.xlim([-0.035, 0.07])
plt.ylim([-0.065, 0.07])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda y, _:
                                                  '{:.0%}'.format(y)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _:
                                                  '{:.0%}'.format(y)))
plt.xlabel('$R_{1}$')
plt.ylabel('$R_{2}$')
legend = plt.legend(['Returns( & sample mean/covariance)',
                     'Black-Litterman prior (equilibrium)',
                     'Black-Litterman posterior', 'Black-Litterman sure',
                     'Min. Rel. Entropy distributional view',
                     'regression/Min.Rel.Entropy point view'])

str1 = r'Confidence level in the prior: $\tau$ = %d ' % np.floor(tau)
str2 = 'Confidence level in the views: c = %d' % np.floor(100 * c)
plt.text(0.036, -0.034, str1)
plt.text(0.036, -0.038, str2 + '%')

plt.grid(True)
add_logo(fig, location=1)
plt.tight_layout()
