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

# # s_dcc_fit [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_dcc_fit&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-inv-extr-dyn-cop).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

from arpym.estimation import exp_decay_fp, fit_garch_fp, fit_dcc_t,\
                             fit_locdisp_mlfp, cov_2_corr, factor_analysis_paf
from arpym.tools import plot_ellipse, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-parameters)

n_ = 40  # number of stocks
t_first = '2009-01-01'  # starting date
t_last = '2012-01-01'  # ending date
k_ = 10  # number of factors
nu = 4.  # degrees of freedom
tau_hl = 120  # prior half life
i_1 = 27  # index of first quasi-invariant shown in plot
i_2 = 29  # index of second quasi-invariant shown in plot

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-implementation-step00): Load data

# +
# upload stocks values
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
df_stocks = pd.read_csv(path + 'db_stocks_sp.csv',  skiprows=[0], index_col=0)

# set timestamps
df_stocks = df_stocks.set_index(pd.to_datetime(df_stocks.index))

# select data within the date range
df_stocks = df_stocks.loc[(df_stocks.index >= t_first) &
                          (df_stocks.index <= t_last)]

# remove the stocks with missing values
df_stocks = df_stocks.dropna(axis=1, how='any')
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-implementation-step01): Compute log-returns

v_stock = np.array(df_stocks.iloc[:, :n_])
dx = np.diff(np.log(v_stock), axis=0)  # S&P 500 index compounded return
t_ = dx.shape[0]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-implementation-step02): Set flexible probabilities

p = exp_decay_fp(t_, tau_hl)  # flexible probabilities

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-implementation-step03): Fit a GARCH(1,1) on each time series of compounded returns

param = np.zeros((4, n_))
sigma2 = np.zeros((t_, n_))
xi = np.zeros((t_, n_))
for n in range(n_):
    param[:, n], sigma2[:, n], xi[:, n] = \
        fit_garch_fp(dx[:, n], p, rescale=True)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-implementation-step04): Estimate marginal distributions by fitting a Student t distribution via MLFP

mu_marg = np.zeros(n_)
sigma2_marg = np.zeros(n_)
for n in range(n_):
    mu_marg[n], sigma2_marg[n] = fit_locdisp_mlfp(xi[:, n], p=p, nu=nu)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-implementation-step05): Map each marginal time series into standard normal realizations

xi_tilde = np.zeros((t_, n_))
for n in range(n_):
    u = t.cdf(xi[:, n], df=10**6, loc=mu_marg[n],
              scale=np.sqrt(sigma2_marg[n]))
    u[u <= 10**(-7)] = 10**(-7)
    u[u >= 1-10**(-7)] = 1-10**(-7)
    xi_tilde[:, n] = t.ppf(u, df=10**6)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-implementation-step06): Estimate the unconditional correlation matrix via MLFP

# +
_, sigma2_xi_tilde = fit_locdisp_mlfp(xi_tilde, p=p, nu=10**6)
rho2_xi_tilde, _ = cov_2_corr(sigma2_xi_tilde)
rho2 = rho2_xi_tilde

beta, delta2 = factor_analysis_paf(rho2_xi_tilde, k_)
rho2 = beta @ beta.T + np.diag(delta2)
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_dcc_fit-implementation-step07): Compute the time series of true invariants via DCC fit

params, r2_t, epsi, q2_t_ = fit_dcc_t(xi_tilde, p, rho2=rho2)
c, a, b = params
q2_t_nextstep = c*rho2 +\
                b*q2_t_ +\
                a*(np.array([epsi[-1, :]]).T@np.array([epsi[-1, :]]))
r2_t_nextstep, _ = cov_2_corr(q2_t_nextstep)

# ## Save the data to temporary databases

path = '../../../databases/temporary-databases/'
df_xi = pd.DataFrame(data=xi, index=df_stocks.index[1:],
                     columns=df_stocks.columns[:n_])
df_xi.to_csv(path + 'db_GARCH_residuals.csv')

# ## Plots

# +
plt.style.use('arpm')

# Scatter plot
xi_plot = xi[:, [i_1, i_2]]
fig = plt.figure()
plt.scatter(xi[:, i_1], xi[:, i_2], 2, marker='o', linewidths=1)
plt.axis('equal')
plt.axis([np.percentile(xi_plot[:, 0], 2), np.percentile(xi_plot[:, 0], 98),
          np.percentile(xi_plot[:, 1], 2), np.percentile(xi_plot[:, 1], 98)])
plt.xlabel('$\Xi_{%1.f}$' % (i_1+1))
plt.ylabel('$\Xi_{%1.f}$' % (i_2+1))
plt.ticklabel_format(style='sci', scilimits=(0, 0))

# Ellipsoids
mu_plot = np.zeros(2)
rho2_plot = rho2[np.ix_([i_1, i_2], [i_1, i_2])]
r2_t_plot = r2_t_nextstep[np.ix_([i_1, i_2], [i_1, i_2])]
ell_unc = plot_ellipse(mu_plot, rho2_plot, color='b')
ell_cond = plot_ellipse(mu_plot, r2_t_plot, color='tomato')

plt.legend(['Unconditional correlation: $rho^{2}$=%1.2f %%' %
            (100*rho2_plot[0, 1]),
            'Conditional correlation: $r^{2}_{t+1}$=%1.2f %%' %
            (100*r2_t_plot[0, 1]),
            'Quasi-invariants'])
plt.title('Dynamic conditional correlation')
add_logo(fig, location=2)
