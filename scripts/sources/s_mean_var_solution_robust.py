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

# # s_mean_var_solution_robust [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_mean_var_solution_robust&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-mean-var-solution-robust).

# +
import numpy as np
import cvxopt
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution_robust-parameters)

# +
n_ = 10  # number of stocks
v_budget = 1000  # budget at time t_now

v_in = 200  # initial variance
v_fin = 5000  # final variance
v_ = 100  # variance grid
p_in = 10**-9  # initial probability
p_fin = 0.25  # final probability
p_ = 10  # probability grid

r_rf = 0.02  # risk-free rate
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution_robust-implementation-step01): Generate current values and P&L expectation and covariance and define robustness matrix

# +
v_tnow = np.random.lognormal(4, 0.05, n_)

mu_pi = 0.5*np.arange(1, n_+1)
sig2_pi = 0.2*np.ones((n_, n_)) + 0.8*np.eye(n_)
sig2_pi = np.diag(mu_pi)@sig2_pi@np.diag(mu_pi)


# robustness matrix is the diagonal matrix of the P&L's variances
t = np.diag(np.diag(sig2_pi))
# high penalty for low-variance P&L's
t[t >= np.median(np.diag(t))] = 10**-5*t[t >= np.median(np.diag(t))]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution_robust-implementation-step02): Spectral decompositions of the matrices sig2_pi, t

lam2_sig2_pi, e_sig2_pi = np.linalg.eig(sig2_pi)
lam2_sig2_pi = np.diag(lam2_sig2_pi)
lam2_t, e_t = np.linalg.eig(t)
lam2_t = np.diag(lam2_t)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution_robust-implementation-step03): Solve the first step of the mean-variance approach

# +
# Constraints:
# 1) budget constraint: h'*v_tnow = v_budget
# 2) no-short-sale: h>=0

v_span = np.linspace(v_in, v_fin, v_)  # variances
p_span = np.linspace(p_in, p_fin, p_)
q_span = np.sqrt(chi2.ppf(p_span, df=n_))  # quantiles

h_lambda = np.zeros((n_, v_, p_))
mu_h_lambda = np.zeros((v_, p_))
sig2_h_lambda = np.zeros((v_, p_))

cvxopt.solvers.options['show_progress'] = False

print('First step of mean-variance approach')
for v in range(v_):
    for q in range(p_):
        # objective
        c_opt = cvxopt.matrix(np.r_[1, -mu_pi], tc='d')

        # equality constraints: budget
        A_opt = cvxopt.matrix(np.r_[0, v_tnow], size=(1, n_+1), tc='d')
        b_opt = cvxopt.matrix(v_budget, tc='d')

        # inequality constraints
        # no-short-sale
        Gl_opt = cvxopt.matrix(np.block([[0, np.zeros((1, n_))],
                                         [np.zeros((n_, 1)), -np.eye(n_)]]))
        hl_opt = cvxopt.matrix(np.zeros((n_+1)))
        # variance
        Gq0_opt = cvxopt.matrix(np.block([[0, np.zeros((1, n_))],
                                          [np.zeros((n_, 1)),
                                           -np.sqrt(lam2_sig2_pi) @
                                           e_sig2_pi.T]]))
        hq0_opt = cvxopt.matrix(np.r_[np.sqrt(v_span[v]), np.zeros(n_)])
        # robustness
        Gq1_opt = cvxopt.matrix(np.block([[-1, np.zeros((1, n_))],
                                          [np.zeros((n_, 1)),
                                           -q_span[q] *
                                           np.sqrt(lam2_t)@e_t.T]]))
        hq1_opt = cvxopt.matrix(np.zeros(n_+1))

        Gq_opt = [Gq0_opt, Gq1_opt]
        hq_opt = [hq0_opt, hq1_opt]

        # solve
        prob = cvxopt.solvers.socp(c=c_opt,
                                   Gl=Gl_opt, hl=hl_opt,
                                   Gq=Gq_opt, hq=hq_opt,
                                   A=A_opt, b=b_opt)

        if prob['x'] is not None:
            h_lambda[:, v, q] = np.array(prob['x'])[1:, 0]
        else:
            print('\nInfeasible problem for parameters:\n')
            print('v = ' + str(v_span[v]) + '  ' + 'p = ' + str(p_span[q]))

        # Compute the efficient frontier
        mu_h_lambda[v, q] = h_lambda[:, v, q]@mu_pi
        sig2_h_lambda[v, q] = h_lambda[:, v, q].T @\
            sig2_pi @\
            h_lambda[:, v, q]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution_robust-implementation-step03): Compute weights

w_lambda = (h_lambda.T*v_tnow).T / v_budget

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_mean_var_solution_robust-implementation-step04): Solve the second step of the mean-variance approach

# +
print('Second step of mean-variance approach')

# satisfaction = Sharpe ratio
satis_h_lambda = mu_h_lambda / np.sqrt(sig2_h_lambda)

# optimal variance and robustness penalty
lambda_star_ind = np.where(satis_h_lambda == satis_h_lambda.max())
v_star_ind = lambda_star_ind[0][0]
q_star_ind = lambda_star_ind[1][0]
v_star = v_span[v_star_ind]
q_star = q_span[q_star_ind]
# optimal holdings and weights
h_qsi_star = h_lambda[:, v_star_ind, q_star_ind]
w_qsi_star = w_lambda[:, v_star_ind, q_star_ind]
# -

# ## Plots

# +
plt.style.use('arpm')

x0 = max(np.sqrt(sig2_h_lambda[:, 0]).min(),
         np.sqrt(sig2_h_lambda[:, -1]).min())
x1 = min(np.sqrt(sig2_h_lambda[:, 0]).max(),
         np.sqrt(sig2_h_lambda[:, -1]).max())
xlim = [x0, x1]

fig = plt.figure()

# Non-robust
ax11 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=1)
plt.plot(np.sqrt(sig2_h_lambda[:, 0]),
         mu_h_lambda[:, 0])
plt.plot(np.sqrt(sig2_h_lambda[v_star_ind, 0]),
         mu_h_lambda[v_star_ind, 0],
         '.', markersize=15, color='k')
plt.legend(['Efficient frontier', 'Optimal holdings'])
plt.ylabel('$E\{Y_{h}\}$')
plt.xlabel('$Sd\{Y_{h}\}$')
plt.xlim(xlim)
str_opt = '$p =$ %1.2f %%' % np.float(100*p_span[0])
plt.text(0.8, 0.1, str_opt, horizontalalignment='center',
         verticalalignment='center', transform=ax11.transAxes)
plt.title('Non-robust mean-variance efficient frontier', fontweight='bold')

ax12 = plt.subplot2grid((2, 4), (1, 0), colspan=2, rowspan=2)
colors = cm.get_cmap('Spectral')(np.arange(n_)/n_)[:, :3]
for n in range(n_):
    if n == 0:
        plt.fill_between(np.sqrt(sig2_h_lambda[:, 0]),
                         w_lambda[n, :, 0],
                         np.zeros(v_), color=colors[n, :])
    else:
        plt.fill_between(np.sqrt(sig2_h_lambda[:, 0]),
                         np.sum(w_lambda[:n+1, :, 0], axis=0),
                         np.sum(w_lambda[:n, :, 0], axis=0),
                         color=colors[n, :])
plt.axvline(x=np.sqrt(sig2_h_lambda[v_star_ind, 0]), color='k')
plt.ylabel('$w$')
plt.xlabel('$Sd\{Y_{h}\}$')
plt.xlim(xlim)
plt.ylim([0, 1])
plt.title('Non-robust portfolio weights', fontweight='bold')

plt.tight_layout()

# Robust
ax21 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=1)
plt.plot(np.sqrt(sig2_h_lambda[:, -1]),
         mu_h_lambda[:, -1])
plt.plot(np.sqrt(sig2_h_lambda[v_star_ind, -1]),
         mu_h_lambda[v_star_ind, -1],
         '.', markersize=15, color='k')
plt.legend(['Efficient frontier', 'Optimal holdings'])
plt.ylabel('$E\{Y_{h}\}$')
plt.xlabel('$Sd\{Y_{h}\}$')
plt.xlim(xlim)
str_opt = '$p =$ %1.2f %%' % np.float(100*p_span[-1])
plt.text(0.8, 0.1, str_opt, horizontalalignment='center',
         verticalalignment='center', transform=ax21.transAxes)
plt.title('Robust mean-variance efficient frontier', fontweight='bold')
add_logo(fig, axis=ax21, location=5, size_frac_x=1/8)
plt.tight_layout()

ax22 = plt.subplot2grid((2, 4), (1, 2), colspan=2, rowspan=1)
colors = cm.get_cmap('Spectral')(np.arange(n_)/n_)[:, :3]
for n in range(n_):
    if n == 0:
        plt.fill_between(np.sqrt(sig2_h_lambda[:, -1]),
                         w_lambda[n, :, -1],
                         np.zeros(v_), color=colors[n, :])
    else:
        plt.fill_between(np.sqrt(sig2_h_lambda[:, -1]),
                         np.sum(w_lambda[:n+1, :, -1], axis=0),
                         np.sum(w_lambda[:n, :, -1], axis=0),
                         color=colors[n, :])
plt.axvline(x=np.sqrt(sig2_h_lambda[v_star_ind, -1]), color='k')
plt.ylabel('$w$')
plt.xlabel('$Sd\{Y_{h}\}$')
plt.xlim(xlim)
plt.ylim([0, 1])
plt.title('Robust portfolio weights', fontweight='bold')

plt.tight_layout()
