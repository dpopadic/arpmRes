#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"jupyter": {"source_hidden": true}, "cell_type": "markdown"}
# # s_fit_yields_var1 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fit_yields_var1&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from arpym.tools import plot_ellipse, add_logo
from arpym.estimation import fit_var1
from arpym.pricing import ytm_shadowrates
from arpym.estimation import exp_decay_fp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1-parameters)

tau_select = np.array([2, 5])  # selected times to maturity (years)
yields = True  # true if using yields or false if using shadow rates
tau_hl = 180  # half-life parameter (days)
nu = 4  # degrees of freedom used in VAR(1) fit
t_start = '01-Jul-2002'  # starting date
t_end = '02-Jan-2008'  # ending date

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1-implementation-step00): Load data

tau = np.array([1, 2, 3, 5, 7, 10, 15, 30])  # times to maturity
path = '../../../databases/global-databases/fixed-income/db_yields'
y_db = pd.read_csv(path + '/data.csv', header=0, index_col=0)
y = y_db[tau.astype(float).astype(str)][t_start:t_end].values
t_ = y.shape[0]  # length of the time series of rolling values

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1-implementation-step01): Realized risk drivers (yield or shadow rates)

if yields:
    x = y[:]
else:
    x = ytm_shadowrates(y, eta=0.013)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1-implementation-step02): Flexible probabilities

p = exp_decay_fp(t_, tau_hl)  # exponential decay

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1-implementation-step03): Perform VAR(1) fit

b_hat, mu_epsi_hat, sig2_epsi_hat = fit_var1(x, p, nu=nu)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1-implementation-step04): Recovered values of the risk drivers from the fit

x_fit = x[-1, :] @ b_hat.T + mu_epsi_hat.reshape((1, -1))

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1-implementation-step05): Expectation and covariance of the conditional next step prediction

ind = (np.array([np.where(tau == tau_select[i])[0]
                 for i, item in enumerate(tau_select)]).reshape((-1)))
# next-step expectation for all times to maturity
mu_ns = x[-1, :] @ b_hat.T + mu_epsi_hat
# next-step expectation for selected times to maturity
mu_select_ns = mu_ns[ind]
# next-step covariance for all times to maturity
sig2_ns = sig2_epsi_hat
# next-step covariance for selected times to maturity
sig2_select_ns = sig2_ns[np.ix_(ind, ind)]

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yields_var1-implementation-step06): Save databases

# +
out = pd.DataFrame({tau[i]: x[:, i] for i in range(len(tau))})
out['mu_epsi'] = pd.Series(mu_epsi_hat)
out['sig2_epsi'] = pd.Series(sig2_epsi_hat.flatten())
out['b'] = pd.Series(b_hat.flatten())
out['t_now'] = '02-Jan-2008'

if yields:
    out.to_csv('../../../databases/temporary-databases/db_yield_var1_fit.csv',
               index=None)
else:
    out.to_csv('../../../databases/temporary-databases/db_shadowrate_var1_fit.csv',
               index=None)

del out
# -

# ## Plots

# +
plt.style.use('arpm')

x_fit = x_fit.reshape(-1)
fig1 = plt.figure()
plt.plot(tau, x_fit, markersize=15, color='b')
plt.plot(tau, x[-1, :], markersize=15, color=[1, 0.6, 0],
         marker='.', linestyle='none')
plt.xlim([np.min(tau) - 0.2, np.max(tau) + 0.2])
plt.ylim([np.min(x_fit) - 0.001, np.max(x_fit) + 0.001])
plt.xlabel('Time to Maturity (years)')
plt.ylabel('Shadow rate')
plt.xticks()
plt.yticks()
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda z, _:
                                    '{:.2%}'.format(z)))
plt.legend(['Fitted', 'Current'])
plt.grid(True)
add_logo(fig1)

# scatter plot of shadow rates for the selected maturities
fig2 = plt.figure()
plt.plot(x[:, ind[0]], x[:, ind[1]], markersize=5,
         color=[0.55, 0.55, 0.55], marker='.', linestyle='none')
xlab = '%2dy rate' % (tau[ind[1]])
ylab = '%2dy rate' % (tau[ind[0]])
plt.ylabel(xlab)
plt.xlabel(ylab)
x_min = np.floor(min(x[:, ind[0]])*100) / 100
x_max = np.ceil(max(x[:, ind[0]])*100) / 100
y_min = np.floor(min(x[:, ind[1]])*100) / 100
y_max = np.ceil(max(x[:, ind[1]])*100) / 100
x_lim = ([x_min, x_max])
y_lim = ([y_min, y_max])
plt.xticks()
plt.yticks()
plt.grid(True)

# next-step ellipsoid
plt.plot([x[-1, ind[0]], x[-1, ind[0]]],
         [x[-1, ind[1]], x[-1, ind[1]]], color=[1, 0.6, 0],
         marker='.', markersize=8, linestyle='none')
plot_ellipse(mu_select_ns, sig2_select_ns, r=2.4, plot_axes=0,
             plot_tang_box=0, color='b', line_width=1.5)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y,
                                    _: '{:.0%}'.format(y)))
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda y,
                                    _: '{:.2%}'.format(y)))
plt.legend(['Past observations', 'Current observation',
            'Next-step prediction'])
add_logo(fig2)
# -


