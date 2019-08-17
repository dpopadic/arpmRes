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

# # s_execution_opt_satisfaction_quantile [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_execution_opt_satisfaction_quantile&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-plopt_-liquidation-satisfaction).

# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_execution_opt_satisfaction_quantile-parameters)

q_now = 0  # starting volume time
q_end = 1  # ending volume time
h_q_now = 100  # initial holdings
h_q_end = 90  # final holdings
eta = 0.135  # transation price dynamics parameters
sigma = 1.57
lam = np.arange(0.01, 1, 0.05)  # mean-variance trade-off penalties
c = 0.95  # confidence level
k_ = 721  # number of grid points in [q_now, q_end)

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_execution_opt_satisfaction_quantile-implementation-step00): Define grid in which Almgren-Chriss trajectories are calculated

q_grid = np.linspace(q_now, q_end, k_)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_execution_opt_satisfaction_quantile-implementation-step01): Compute the trajectory, the trading rate (speed), the P&L mean and variance, the quantile-based index of satisfaction

# +
# initializations
l_ = len(lam)
variance_pl = np.zeros(l_)
mean_pl = np.zeros(l_)
quantile = np.zeros(l_)
traj = np.zeros((k_, l_))

const = np.sqrt(lam / eta)*sigma

for l in range(l_):
    def trajectory(q): return(h_q_now-h_q_end)*np.sinh((const[l])*(q_end-q)) /\
                              np.sinh((const[l])*(q_end-q_now))+h_q_end

    def trajectory2(q): return((h_q_now-h_q_end)*np.sinh((const[l])*(q_end-q))
                               / np.sinh((const[l])*(q_end-q_now))+h_q_end)**2

    def speed2(q): return (-const[l]*(h_q_now-h_q_end)*np.cosh((const[l]) *
                           (q_end-q)) / np.sinh((const[l])*(q_end-q_now)))**2
    mean_pl[l] = -eta*quad(speed2, q_now, q_end)[0]
    variance_pl[l] = sigma**2*quad(trajectory2, q_now, q_end)[0]
    quantile[l] = norm.ppf(1-c, mean_pl[l],
                           np.sqrt(variance_pl[l]))
    traj[:, l] = trajectory(q_grid)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_execution_opt_satisfaction_quantile-implementation-step02): Find the value of lam that maximizes the satisfaction.

lambda_star = lam[quantile == np.max(quantile)]  # optimal lambda

# ## Save data

# +
output = {'k_': pd.Series(k_),
          'l_': pd.Series(l_),
          'h_q_now': pd.Series(h_q_now),
          'h_q_end': pd.Series(h_q_end),
          'lam': pd.Series(lam),
          'lambda_star': pd.Series(lambda_star),
          'q_grid': pd.Series(q_grid),
          'traj': pd.Series(traj.reshape((k_*l_,)))}

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/\
db_execution_opt_satisfaction_quantile.csv')
# -

# ## Plots

# +
plt.style.use('arpm')

fig, ax = plt.subplots(2, 1)
# plot of the optimal trading trajectory and suboptimal trajectories
lgrey = [0.6, 0.6, 0.6]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey

rand_indexes = np.arange(0, l_)
rr_ = len(rand_indexes)

plt.sca(ax[0])
for r in range(rr_):
    p1 = plt.plot(np.r_[q_now, q_grid],
                  np.r_[h_q_now, traj[:, rand_indexes[r]]],
                  color=lgrey,
                  label='Suboptimal trajectories on the M-V frontier')

p2 = plt.plot(q_grid, traj[:, lam == lambda_star].flatten(), color='r', lw=1.2,
              label='Optimal trajectory $\lambda$ =  % 2.2f' % lambda_star)
plt.ylim([h_q_end - 2, h_q_now + 2])
plt.xlabel('Time')
plt.ylabel('Holdings')
plt.title('Optimal trajectory in the Almgren-Chriss model')
plt.legend(handles=[p1[0], p2[0]])

# plot of the mean-variance frontier
plt.sca(ax[1])
plt.plot(variance_pl[rand_indexes], mean_pl[rand_indexes], '.',
         color=dgrey, markersize=10)
plt.plot(variance_pl[lam == lambda_star], mean_pl[lam == lambda_star], '.',
         color='r', markersize=15)
plt.ylabel('mean')
plt.xlabel('variance')
plt.title('Mean-Variance frontier')
add_logo(fig)
plt.tight_layout()
