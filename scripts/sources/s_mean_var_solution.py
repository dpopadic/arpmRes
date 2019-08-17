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

# # s_mean_var_solution [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_mean_var_solution&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-mean-var-solution).

# +
import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution-parameters)

n_ = 10  # number of stocks
v_budget = 1000  # budget at time t_now
lambda_in = 0  # initial value for the mean-variance penalty
lambda_fin = 1  # final value for the mean-variance penalty
lambda_ = 100  # number of points in the efficient frontier
r_rf = 0.02  # risk-free rate

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution-implementation-step01): Generate current values and P&L expectation and covariance

# +
v_tnow = np.random.lognormal(4, 0.05, n_)

mu_pi = 0.5*np.arange(1, n_+1)
sig2_pi = 0.2*np.ones((n_, n_)) + 0.8*np.eye(n_)
sig2_pi = np.diag(mu_pi)@sig2_pi@np.diag(mu_pi)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution-implementation-step02): Solve the first step of the mean-variance approach

# +
# Constraints:
# 1) budget constraint: h'*v_tnow = v_budget
# 2) no-short-sale: h>=0

lambda_span = np.linspace(lambda_in, lambda_fin, lambda_)
h_lambda = np.zeros((n_, lambda_))

cvxopt.solvers.options['show_progress'] = False
for l in range(lambda_):
    # objective
    P_opt = cvxopt.matrix(2*lambda_span[l]*sig2_pi)
    q_opt = cvxopt.matrix(-(1-lambda_span[l])*mu_pi)
    # inequality constraints: no-short-sale
    G_opt = cvxopt.matrix(-np.eye(n_))
    h_opt = cvxopt.matrix(np.zeros(n_))
    # equality constraints: budget
    A_opt = cvxopt.matrix(v_tnow).T
    b_opt = cvxopt.matrix(v_budget, tc='d')
    # solve
    h_lambda[:, l] = np.array(cvxopt.solvers.qp(P_opt, q_opt, G_opt, h_opt,
                                                A_opt, b_opt)['x'])[:, 0]

# efficient frontier

mu_h_lambda = h_lambda.T@mu_pi - r_rf
sig2_h_lambda = np.diag(h_lambda.T@sig2_pi@h_lambda)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution-implementation-step03): Compute weights

w_lambda = (h_lambda.T*v_tnow).T / v_budget

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution-implementation-step04): Solve the second step of the mean-variance approach

# +
# Satisfaction = Sharpe ratio
satis_h_lambda = mu_h_lambda / np.sqrt(sig2_h_lambda)

# optimal variance and robustness penalty
lambda_star_ind = np.argmax(satis_h_lambda)
lambda_star = lambda_span[lambda_star_ind]
# optimal holdings and weights
h_qsi_star = h_lambda[:, lambda_star_ind]
w_qsi_star = w_lambda[:, lambda_star_ind]
# -

# ## Plots

# +
plt.style.use('arpm')

x0 = np.sqrt(sig2_h_lambda).min()
x1 = np.sqrt(sig2_h_lambda).max()
xlim = [x0, x1]

fig = plt.figure()

ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
plt.plot(np.sqrt(sig2_h_lambda), mu_h_lambda)
plt.plot(np.sqrt(sig2_h_lambda[lambda_star_ind]), mu_h_lambda[lambda_star_ind],
         '.', markersize=15, color='k')
plt.legend(['Efficient frontier', 'Optimal holdings'])
plt.ylabel('$E\{Y_{h}\}$')
plt.xlabel('$Sd\{Y_{h}\}$')
plt.xlim(xlim)
plt.title('Mean-variance efficient frontier', fontweight='bold')
add_logo(fig, axis=ax1)
plt.tight_layout()

ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=4)
colors = cm.get_cmap('Spectral')(np.arange(n_)/n_)[:, :3]
for n in range(n_):
    if n == 0:
        plt.fill_between(np.sqrt(sig2_h_lambda), w_lambda[n, :],
                         np.zeros(lambda_), color=colors[n, :])
    else:
        plt.fill_between(np.sqrt(sig2_h_lambda),
                         np.sum(w_lambda[:n+1, :], axis=0),
                         np.sum(w_lambda[:n, :], axis=0), color=colors[n, :])
plt.axvline(x=np.sqrt(sig2_h_lambda[lambda_star_ind]), color='k')

plt.ylabel('$w$')
plt.xlabel('$Sd\{Y_{h}\}$')
plt.xlim(xlim)
plt.ylim([0, 1])
plt.title('Portfolio weights', fontweight='bold')

plt.tight_layout()
