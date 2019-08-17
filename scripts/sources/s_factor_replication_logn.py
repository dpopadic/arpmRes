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

# # s_factor_replication_logn [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_factor_replication_logn&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-fac-rep-port-log-norm).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from arpym.statistics import simulate_normal, multi_r2
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_factor_replication_logn-parameters)

n_max = 500  # max target dimension
n_step = np.arange(25, n_max + 25, 25)  # target dimension grid
j_ = 10000  # number of scenarios
mu = np.append(1, np.zeros(n_max))
delta = np.random.rand(n_max)
sigma2 = np.diag(np.append(1, delta**2))
c = np.exp(mu+np.diag(sigma2)/2)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_factor_replication_logn-implementation-step01): Choose arbitrary parameters

alpha = np.zeros(n_max)
beta = simulate_normal(np.zeros(1), np.eye(1), n_max).reshape(-1, 1)  # generate normal scenarios

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_factor_replication_logn-implementation-step02): Compute scenarios of the factor, residuals and target variables

h = np.random.lognormal(mu[0], sigma2[0, 0], size=(j_, 1)) - c[0]
l = simulate_normal(np.zeros(n_max), np.eye(n_max), j_).reshape(-1, n_max)
u = np.exp(l * delta) - np.exp(delta ** 2 / 2.)
x = alpha + h @ beta.T + u

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_factor_replication_logn-implementation-step03): Compute expectation and covariance of the target variables

mu_x = alpha
sigma2_x = beta @ beta.T + np.diag(delta ** 2)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_factor_replication_logn-implementation-step04): Compute extraction matrix and r-squared

beta_ = beta.T * (delta ** -2)
var_h = np.exp(3-2*np.exp(1.5))*(np.exp(1)-1)
r2 = np.zeros(len(n_step))
for i, n in enumerate(n_step):
    beta_inv = np.linalg.solve(beta_[:, :n] @ beta[:n, :], beta_[:, :n])
    sigma2_z_h = (beta_inv * delta[:n]) @ (beta_inv * delta[:n]).T
    r2[i] = multi_r2(sigma2_z_h, np.atleast_2d(var_h))

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_factor_replication_logn-implementation-step05): Compute cross-sectional factor and premia

z_cs = x @ beta_inv.T
lam = beta_inv @ alpha

# ## Plots

# +
plt.style.use('arpm')

f = plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)

gs1 = GridSpec(3, 3)
gs1.update(left=0.05, right=0.48, wspace=0.2)
ax1 = plt.subplot(gs1[:-1, 0])
plt.hist(z_cs[:, 0], bins=int(30*np.log(j_)),
         orientation='horizontal', bottom=0)
plt.gca().invert_xaxis()
ax1.tick_params(labelsize=12)

ax2 = plt.subplot(gs1[:-1, 1:])
plt.scatter(h[:, 0], z_cs[:, 0], marker='.', s=0.5)
plt.scatter(0, lam, marker='.', color='r', s=50)
plt.ylim([-10, 200])
ax2.tick_params(axis='x', colors='None')
ax2.tick_params(axis='y', colors='None')
plt.xlim([-10, 100])
plt.xlabel('$H$', labelpad=-16)
plt.ylabel('$Z^{CS}$', labelpad=-20)
plt.title('Scatter plot for n = %d' % n_max)
plt.legend(['sample', 'expectation'])
ax3 = plt.subplot(gs1[-1, 1:])
plt.hist(h[:, 0], bins=int(120*np.log(j_)), bottom=0)
ax3.tick_params(labelsize=12)
plt.gca().invert_yaxis()

gs2 = GridSpec(3, 3)
gs2.update(left=0.55, right=0.98, hspace=0.05)
ax4 = plt.subplot(gs2[:-1, :])
plt.plot(n_step, r2, 'r', lw=1.2)
plt.plot([0, n_max], [1, 1], 'b', lw=2)
plt.xlabel('target dimension')
plt.ylabel('r-square')
plt.title('Factor-replicating portfolio convergence')

add_logo(f, location=4, set_fig_size=False)
