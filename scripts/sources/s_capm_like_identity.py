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

# # s_capm_like_identity [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_capm_like_identity&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-capmlike-lin-copy-1).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.statistics import meancov_sp, simulate_normal
from arpym.estimation import cov_2_corr
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-parameters)

# +
n_ = 100  # number of financial instruments
j_ = 10000  # number of simulations
v_budget = 5  # current budget
r_tnow_thor = 0.02  # risk-free interest rate
v_tnow = np.ones(n_)  # current values
sigma_mu = 1 / 30  # scale of compounded returns' expectation
sigma_bar = 1 / 40  # scale of compounded returns' covariance
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-implementation-step01): Generate the parameters of the compounded returns distribution

# +
mu = simulate_normal(np.zeros(n_), sigma_mu**2*np.eye(n_), 1)  # expectation of compunded returns
a = np.zeros((n_, n_))
for n in range(n_):
    a[n, :] = simulate_normal(np.zeros(n_), sigma_bar**2*np.eye(n_), 1)
sigma2 = (a@a.T)  # covariance of compounded returns
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-implementation-step02): Generate the MC scenarios of the compounded returns

# +
c_tnow_thor = simulate_normal(mu, sigma2, j_)  # compounded returns scenarios
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-implementation-step03): Compute the scenarios of the linear returns

# +
# linear returns scenarios
r_tnow_thor_j = np.exp(c_tnow_thor) - 1
# linear returns expectation and covariance
mu_r, sigma2_r = meancov_sp(r_tnow_thor_j)
# correlation and volatility vector
c2_r, sigmavol_r = cov_2_corr(sigma2_r)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-implementation-step04): Compute the MC scenarios of P&L's

# +
# P&L scenarios
pi_tnow_thor = r_tnow_thor_j * v_tnow
# P&L expectation and covariance
mu_pi, sigma2_pi = meancov_sp(pi_tnow_thor)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-implementation-step05): Compute the maximum Sharpe ratio portfolio

# +
# maximum Sharpe ratio portfolio
h_sr = (v_budget / (v_tnow.T@np.linalg.solve(sigma2_pi, mu_pi - r_tnow_thor*v_tnow))) * \
    np.linalg.inv(sigma2_pi)@(mu_pi - r_tnow_thor*v_tnow)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-implementation-step06): Compute the scenarios of the max. Sharpe ratio portfolio return

# +
# maximum Sharpe ratio portfolio return
r_sr = (pi_tnow_thor@h_sr) / v_budget
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-implementation-step07): Compute the left and the right hand side of the CAPM-like identity

# +
# left hand side
y = mu_pi - r_tnow_thor*v_tnow
# right hand side
mu_r_sr, sigma2_r_sr = meancov_sp(r_sr)
_, sigma2_pi_r_sr = meancov_sp(np.concatenate((pi_tnow_thor, np.atleast_2d(r_sr).T),
                                             axis=1))
beta = sigma2_pi_r_sr[:-1, -1]/sigma2_r_sr
x = beta*(mu_r_sr - r_tnow_thor)  # right hand side
# -

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_capm_like_identity-implementation-step08): Compute the scenarios of the residuals of the linear factor model

# +
# compute residuals
u = (pi_tnow_thor - r_tnow_thor*v_tnow) - np.outer(r_sr - r_tnow_thor, beta)
# covariance of the residuals
_, sigma2_u = meancov_sp(u)
# correlation of the residuals
sigma_u, _ = cov_2_corr(sigma2_u)
# -

# ## Plots

# +
plt.style.use('arpm')

# Visualize the security market line

fig1 = plt.figure()
xx = np.linspace(np.min(x), np.max(x), 2)
plt.plot(x, y, '.', markersize=10)
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.plot(1.05*xx, 1.05*xx, color=[0.4, 0.4, 0.4])
plt.xlabel(r'$\frac{Cv\{\mathbf{\Pi}, R^{SR}\}}{V\{R^{SR}\}} (E[R^{SR}]-r_{t_{now}\rightarrow t_{hor}})$')
plt.ylabel(r'$E[\mathbf{\Pi}]-r_{t_{now}\rightarrow t_{hor}}\mathbf{v}$')
plt.title('Security market line')
plt.grid(True)
add_logo(fig1)
plt.tight_layout()

# Plot the correlation matrix heat of returns

fig2 = plt.figure()
hcorrel = plt.imshow(c2_r)
plt.colorbar(hcorrel)
plt.title('Correlation of linear returns')
add_logo(fig2, size_frac_x=1/4, location=9, alpha=1.0)
plt.tight_layout()

# Plot the histogram of the off-diagonal elements of the residuals correlation
# matrix

# Extrapolate the off-diagonal elements
elem = sigma_u[np.triu_indices(n_, k=1)]

fig3 = plt.figure()
# set uniform probabilities
p2 = np.ones(elem.shape[0]) / elem.shape[0]
# compute histogram
h, b = histogram_sp(elem, p=p2, k_=40)
plt.bar(b, h, width=b[1]-b[0])
plt.title('Off-diagonal correlation of residuals')
add_logo(fig3)
plt.tight_layout()

# Plot the vector containing the sorted st.dev of instruments returns and
# the corresponding expectations

fig4, ax = plt.subplots(2, 1)
plt.sca(ax[0])
mean_std = np.stack((mu_r, sigmavol_r), axis=1)
ordr, ordc = np.sort(mean_std[:, 1]), np.argsort(mean_std[:, 1])
# Sorted standard deviations
sorted_meanStd = mean_std[ordc, :]
plt.bar(np.arange(n_), sorted_meanStd[:, 0], width=1)
plt.axis([0, (n_ - 0.5), 1.07*np.min(np.mean(r_tnow_thor_j, 0)),
          1.1*np.max(np.mean(r_tnow_thor_j, 0))])
plt.title('Expectation of linear returns')

plt.sca(ax[1])
plt.bar(np.arange(n_), sorted_meanStd[:, 1], width=1)
plt.axis([0, (n_ - 0.5), 1.05*np.min(np.mean(r_tnow_thor_j, 0)), np.max(np.std(r_tnow_thor_j, 0))])
plt.title('St.dev of linear returns')
add_logo(fig4, location=2)
plt.tight_layout()

# Dollars investment in each instrument to obtain the max. Sharpe ratio
# portfolio

fig5 = plt.figure()
y = v_tnow * h_sr
plt.bar(np.arange(n_), y, width=1)
plt.axis([0, (n_ - 0.5), 1.05*np.min(v_tnow * h_sr),
          1.05*np.max(v_tnow * h_sr)])
plt.ylabel('Investment $')
plt.title('Max Sharpe ratio portfolio')
add_logo(fig5)
plt.tight_layout()
