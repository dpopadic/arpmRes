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

# # s_aggregation_regcred [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_aggregation_regcred&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-credit-reg-fram).

# +
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd

from arpym.tools import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_regcred-parameters)

n_ = 1000  # number of bonds in the portfolio
ll = 2.5 * 1e6  # amount of losses
a_z, b_z, i_ = -4.5, 4.5, 21  # boundaries, size of grid for the risk factor
j_ = 1000  # number of scenarios of the P&L at the horizon
i_star = 9  # selected index for the realization of the variable z
a_c, b_c, k_ = 0.001, 0.999, 1000  # boundaries, size of the conf. levels grid

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_regcred-implementation-step01): Generate losses, correlations, default probabilities, idyosincratic shocks

loss_n = ll * (0.2*np.random.rand(n_) + 0.9)  # losses
rho_n = 0.8 * np.random.rand(n_)  # correlations
p_n = 0.2 * (0.2*np.random.rand(n_) + 0.9)  # default probabilities
inv_phi_utilde = np.random.randn(j_, n_)  # idiosyncratic shocks

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_regcred-implementation-step02): Compute pdf of the conditional distribution.

# +
z = np.linspace(a_z, b_z, i_)  # grid of values for the risk factor z
# initializations
z_ = len(z)
e = np.zeros(z_)  # conditional expectation

for i, z_i in enumerate(z):
    aux = (norm.ppf(p_n) - np.sqrt(rho_n) * z_i) / \
        np.sqrt(1 - rho_n)
    e[i] = loss_n @ norm.cdf(aux)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_regcred-implementation-step03): Compute pdf of the unconditional distribution of the the losses

# +
p_j = np.ones(j_) / j_  # probabilities associated to the losses scenarios
def_loss_z = np.zeros(j_)  # conditional losses

# conditional losses
z_n = np.sqrt(rho_n) * z[i_star] + inv_phi_utilde * np.sqrt(1 - rho_n)
indicator_d_n = (z_n <= norm.ppf(p_n)).astype(float)
def_loss_z = loss_n @ indicator_d_n.T
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_regcred-implementation-step04): Compute pdf of the unconditional distribution.

# +
# grid of confidence levels for the loss quantile
c = np.linspace(a_c, b_c, k_)
# initializations
num_grid = 200
# grid of values for the losses
def_loss_grid = np.linspace(np.min(e), np.max(e), num_grid)

q_def_loss = np.zeros(k_)  # quantile of the losses
cdf_def_loss = np.zeros(num_grid)  # approximated cdf of the losses
pdf_def_loss = np.zeros(num_grid)  # approximated pdf of the losses

# unconditional distribution
for k in range(k_):
    aux = (norm.ppf(p_n) - np.sqrt(rho_n) *
           norm.ppf(c[k])) / np.sqrt(1 - rho_n)
    q_def_loss[k] = loss_n.T@norm.cdf(aux)

interp = CubicSpline(np.sort(q_def_loss), c, extrapolate='bool')

cdf_def_loss = interp(def_loss_grid)
pdf_def_loss = np.diff(np.r_['-1', [0], cdf_def_loss]) / \
                (def_loss_grid[1] - def_loss_grid[0])
# -

# ## Save the data

# +
output = {'loss_n': pd.Series(loss_n),
          'p_j': pd.Series(p_j),
          'p_n': pd.Series(p_n),
          'rho_n': pd.Series(rho_n),
          'inv_phi_utilde': pd.Series(inv_phi_utilde.reshape((j_*n_,)))}

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_aggregation_regcred.csv')
# -

# ## Plots

# +
plt.style.use('arpm')

n_bins = 350  # number of bins
x = np.zeros(n_bins)
y = np.zeros(n_bins)

# histogram of the conditional losses
y, x = histogram_sp(def_loss_z, p=p_j, k_=n_bins)

fig = plt.figure()
l_0 = plt.bar(x, y / np.max(y),
              width=np.diff(x, 1)[0], label='Conditional distribution')
l_1 = plt.plot(def_loss_grid, pdf_def_loss / np.sum(pdf_def_loss) * 10,
               'k-', label='Unconditional distribution')
l_2 = plt.plot(e[i_star], 0, 'ro', markersize=5, markeredgecolor='r',
               markerfacecolor='r', label='Conditional expectation')
plt.legend(loc=1, fontsize=14)
plt.ylim([0, 1.5])

plt.title('Regulatory credit'
          ' approximation for {n_counterparties} counterparties'.
          format(n_counterparties=n_))

add_logo(fig)
plt.tight_layout()
