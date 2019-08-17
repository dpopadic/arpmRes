#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_checklist_scenariobased_step10 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_scenariobased_step10&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-10).

# +
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from arpym.statistics import meancov_sp
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step10-parameters)

# +
# indicates which projection to continue from
# True: use copula-marginal projections
# False: use historical projections
copula_marginal = True

q_now = 0  # initial volume time
q_end = 1  # final volume time
k_ = 300  # number of elements in the q grid
l_ = 500  # number of elements in the beta grid
alpha = 1  # parameter of the power slippage component
gamma = 3.14e-5 # permanent impact parameter
eta = 1.42e-6  # temporary impact parameter
c = 0.95  # confidence level for quantile satisfaction measure

n_plot = 2  # index of instrument to plot
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step10-implementation-step00): Load data

# +
path = '../../../databases/temporary-databases/'

# Risk drivers identification
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools.csv')
n_stocks = int(db_riskdrivers_tools['n_stocks'][0])
n_bonds = int(db_riskdrivers_tools.n_bonds[0])
n_ = n_stocks+n_bonds+3

# Aggregation
db_holdings = pd.read_csv(path+'db_holdings.csv')
h = np.squeeze(db_holdings.values)  # initial holdings

if copula_marginal:
    # Projection
    db_scenprob = pd.read_csv(path+'db_scenario_probs.csv')
    p = db_scenprob.p.values

    # Pricing
    # import daily P&Ls computed in step 5 with m_=1
    db_pi_oneday = pd.read_csv(path+'db_oneday_pl.csv')
    pi_oneday = db_pi_oneday.values

    # Construction
    db_final_portfolio = pd.read_csv(path+'db_final_portfolio.csv')
    # the final portfolio is the one obtained in the construction step,
    # that optimizes the cVaR satisfaction measure
    h_qsi = np.squeeze(db_final_portfolio.values)

else:
    # Projection
    db_scenprob = pd.read_csv(path+'db_scenario_probs_bootstrap.csv')
    p = db_scenprob.p.values

    # Pricing
    # import daily P&Ls computed in step 5 with m_=1
    db_pi_oneday = pd.read_csv(path+'db_oneday_pl_historical.csv')
    pi_oneday = db_pi_oneday.values

    # Construction
    db_final_portfolio = pd.read_csv(path+'db_final_portfolio_historical.csv')
    # the final portfolio is the one obtained in the construction step,
    # that optimizes the cVaR satisfaction measure
    h_qsi = np.squeeze(db_final_portfolio.values)

# start portfolio
h_qnow = h
# final portfolio
h_qend = h_qsi
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step10-implementation-step01): Find trajectory

# +
sigma2 = np.zeros(n_)
variance_pihat = np.zeros((n_, l_))
mean_pihat = np.zeros((n_, l_))
xi = np.zeros(l_)
traj = np.zeros((n_, l_, k_))

# size of parent order
delta_h_parent = (h_qend - h_qnow).astype('int')
# beta grid
beta = np.linspace(alpha/(1+alpha), 1, l_+1, endpoint=True)
beta = beta[1:]
# q grid
q_grid = np.linspace(q_now, q_end, k_)

for n in range(n_):
    if delta_h_parent[n] == 0:
        # no change in holdings
        traj[n, :, :] = np.tile(h_qend[n], (l_, k_))
    else:
        _, sigma2[n] = meancov_sp(pi_oneday[:, n], p)
        for l in range(l_):
            # expected P&L
            xi[l] = beta[l]**(alpha+1)/(beta[l]+beta[l]*alpha-alpha)
            mean_pihat[n, l] = gamma/2*(h_qend[n]**2 - h_qnow[n]**2) - \
                eta*xi[l]*np.abs(delta_h_parent[n])**(1+alpha) * \
                (q_end-q_now)**(-alpha)
            # P&L variance
            variance_pihat[n, l] = sigma2[n] * (q_end-q_now) * \
                (h_qnow[n]**2 + 2*h_qnow[n]*delta_h_parent[n]/(beta[l]+1) +
                (delta_h_parent[n]**2)/(2*beta[l]+1))
            # trajectory
            traj[n, l, :] = h_qnow[n] + \
                ((q_grid-q_now)/(q_end-q_now))**beta[l]*delta_h_parent[n]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_scenariobased_step10-implementation-step02): Maximization

q_satis = np.zeros((n_, l_))
beta_star = np.zeros(n_)
l_star = np.zeros(n_)
for n in range(n_):
    if delta_h_parent[n] == 0:
        # no change in holdings
        beta_star[n] = beta[-1]
    else:
        # quantile satisfaction measure
        for l in range(l_):
            q_satis[n, l] = mean_pihat[n, l] + \
                            np.sqrt(variance_pihat[n, l])*norm.ppf(1-c)
        # beta corresponding to the optimal liquidation trajectory
        l_star[n] = \
            np.where(q_satis[n, :] == np.max(q_satis[n, :]))[0]
        beta_star[n] = beta[np.int(l_star[n])]

# ## Plots

# plot execution trajectories
plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)
for i in range(0, l_, 50):
    plt.plot(q_grid, traj[n_plot-1, i, :]*1e-6, color='grey')
plt.plot(q_grid, traj[n_plot-1, np.int(l_star[n_plot-1]), :]*1e-6,
         color='red')
plt.title('Optimal trading trajectory - ' + db_pi_oneday.columns[n_plot-1],
         fontsize=20, fontweight='bold')
plt.xlabel('Volume time', fontsize=17)
plt.ylabel('Holdings (million units)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,1)
add_logo(fig, location=1, set_fig_size=False)
