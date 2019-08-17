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

# # s_projection_var1_yields [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_projection_var1_yields&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_projection_var1_yields).

# +
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from arpym.estimation import var2mvou
from arpym.statistics import simulate_mvou, moments_mvou
from arpym.tools import plot_ellipse, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_projection_var1_yields-parameters)

m_ = 120  # number of monitoring times (proj. hor = m_ months)
deltat_m = 21  # time step (days)
tau_select = np.array([2, 7])  # selected times to maturity (years)
yields = True  # true if using yields or false if using shadow rates
j_ = 1000  # number of Monte Carlo scenarios

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_projection_var1_yields-implementation-step00): Upload data

# upload db from s_fit_shadowrates_var1
path = '../../../databases/temporary-databases'
if yields:
    df = pd.read_csv(path + '/db_yield_var1_fit.csv', header=0)
else:
    df = pd.read_csv(path + '/db_shadowrate_var1_fit.csv', header=0)
tau = np.array([1, 2, 3, 5, 7, 10, 15, 30])
d_ = len(tau)
b = np.array(df['b'].iloc[:d_ ** 2].values.reshape(d_, d_))
mu = np.array(df['mu_epsi'].iloc[:d_])
sig2 = np.array(df['sig2_epsi'].iloc[:d_ ** 2].values.reshape(d_, d_))
t_now = pd.to_datetime(df['t_now'].iloc[0])
t_now = np.datetime64(t_now, 'D')
x = np.array(df[tau.astype('str')])
x_t_ = x[-1, :]

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_projection_var1_yields-implementation-step01): Embedding of VAR(1) into a MVOU process

theta, mu_mvou, sig2_mvou = var2mvou(b, mu, sig2, 1)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_projection_var1_yields-implementation-step02): Monte Carlo scenarios for the MVOU process

x_t_now_t_hor = simulate_mvou(x_t_, np.array([deltat_m
                                              for m in range(1, m_+1)]),
                              theta, mu_mvou, sig2_mvou, j_)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_projection_var1_yields-implementation-step03): Cond. expectations and covariances at the horizon

# +
idx_tau = (np.array([np.where(tau == tau_select[i])[0]
                     for i, item in enumerate(tau_select)]).reshape((-1)))

_, drift_hor, sig2_hor = moments_mvou(x_t_, [m_*deltat_m],
                                      theta, mu_mvou, sig2_mvou)

drift_hor_sel = drift_hor[idx_tau]
sig2_hor_sel = sig2_hor[np.ix_(idx_tau, idx_tau)]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_projection_var1_yields-implementation-step04): Stat. expectation and covariance at the horizon for selected times

# +
_, drift_stat, sig2_stat = moments_mvou(x_t_, np.int64(20 * 252),
                                        theta, mu_mvou, sig2_mvou)

drift_stat_sel = drift_stat[idx_tau]
sig2_stat_sel = sig2_stat[np.ix_(idx_tau, idx_tau)]
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_projection_var1_yields-implementation-step05): Save databases

# +
t_m = np.busday_offset(t_now, np.arange(m_+1)*deltat_m, roll='forward')

output = {}
x_t_hor_save = x_t_now_t_hor.reshape(j_ * (m_+1), 8)
for i, item in enumerate(tau):
    output.update({tau[i]: pd.Series(x_t_hor_save[:, i])})

df = pd.DataFrame(output)
if yields:
    df.to_csv(path+'/db_proj_scenarios_yield.csv',
              index=None)
else:
    df.to_csv(path+'/db_proj_scenarios_shadowrate.csv',
              index=None)

output = {}
output.update({'sig2_mvou': pd.Series(sig2_mvou.reshape(-1))})
output.update({'theta': pd.Series(theta.reshape(-1))})
output.update({'mu_mvou': pd.Series(mu_mvou)})
df = pd.DataFrame(output)
df.to_csv(path+'/db_proj_scenarios_yield_par.csv', index=None)

del df

output = {'dates': pd.Series(t_m)}

df = pd.DataFrame(output)
df.to_csv(path+'/db_proj_dates.csv', index=None)
del df
# -

# ## Plots

# +
# marginal distributions
t_ = 5000  # coarseness of pdfs
x1 = np.zeros((t_, 2))
x2 = np.zeros((t_, 2))
y1 = np.zeros((t_, 2))
y2 = np.zeros((t_, 2))

x1[:, 0] = np.linspace(drift_hor_sel[0] - 4*np.sqrt(sig2_hor_sel[0, 0]),
                       drift_hor_sel[0] + 4*np.sqrt(sig2_hor_sel[0, 0]),
                       t_)
y1[:, 0] = sp.stats.norm.pdf(x1[:, 0], drift_hor_sel[0],
                             np.sqrt(sig2_hor_sel[0, 0]))
x2[:, 0] = np.linspace(drift_hor_sel[1] - 4*np.sqrt(sig2_hor_sel[1, 1]),
                       drift_hor_sel[1] + 4*np.sqrt(sig2_hor_sel[1, 1]),
                       t_)
y2[:, 0] = sp.stats.norm.pdf(x2[:, 0], drift_hor_sel[1],
                             np.sqrt(sig2_hor_sel[1, 1]))

# stationary distributions
x1[:, 1] = np.linspace(drift_stat_sel[0] - 4*np.sqrt(sig2_stat_sel[0, 0]),
                       drift_stat_sel[0] + 4*np.sqrt(sig2_stat_sel[0, 0]),
                       t_)
y1[:, 1] = sp.stats.norm.pdf(x1[:, 1], drift_stat_sel[0],
                             np.sqrt(sig2_stat_sel[0, 0]))
x2[:, 1] = np.linspace(drift_stat_sel[1] - 4*np.sqrt(sig2_stat_sel[1, 1]),
                       drift_stat_sel[1] + 4*np.sqrt(sig2_stat_sel[1, 1]),
                       t_)
y2[:, 1] = sp.stats.norm.pdf(x2[:, 1], drift_stat_sel[1],
                             np.sqrt(sig2_stat_sel[1, 1]))

plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
blue = [0.27, 0.4, 0.9]  # light blue
orange = [0.94, 0.35, 0]  # orange

jsel = 15

fig, ax = plt.subplots(2, 2)

# joint distribution
plt.sca(ax[0, 1])
hs = plt.plot(x_t_now_t_hor[:200, -1, idx_tau[0]],
              x_t_now_t_hor[:200, -1, idx_tau[1]],
              'k.', markersize=3)  # projected scenarios
plot_ellipse(drift_stat_sel, sig2_stat_sel, r=2, plot_axes=0,
             plot_tang_box=0,
             color=dgrey, line_width=1.5)  # stationary ellipsoid
plot_ellipse(drift_hor_sel, sig2_hor_sel, r=2, plot_axes=0,
             plot_tang_box=0,
             color=orange, line_width=1.5)  # selected hor ellipsoid
plt.plot(x[-1, idx_tau[0]], x[-1, idx_tau[1]], marker='o', color='k',
         markerfacecolor='k', markersize=5)  # initial position
plt.plot(drift_hor_sel[0], drift_hor_sel[1], color='g', marker='.',
         markersize=10, markerfacecolor='g')  # mean
plt.grid(True)
plt.xlim([drift_hor_sel[0] - 3 * np.sqrt(sig2_hor_sel[0, 0]),
          drift_hor_sel[0] + 3 * np.sqrt(sig2_hor_sel[0, 0])])
plt.ylim([drift_hor_sel[1] - 3 * np.sqrt(sig2_hor_sel[1, 1]),
          drift_hor_sel[1] + 3 * np.sqrt(sig2_hor_sel[1, 1])])
xlab = '%2d year shadow rate' % (tau[idx_tau[0]])
plt.xlabel(xlab)
ylab = '%2d year shadow rate' % (tau[idx_tau[1]])
plt.ylabel(ylab)
plt.xticks()
plt.yticks()
plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(lambda z, _:
                                    '{:.2%}'.format(z)))
plt.gca().xaxis.set_major_formatter(tck.FuncFormatter(lambda z, _:
                                    '{:.2%}'.format(z)))

# marginal and stationary distributions: bottom plot
plt.sca(ax[1, 1])
plt.xlim([drift_hor_sel[0] - 3 * np.sqrt(sig2_hor_sel[0, 0]),
          drift_hor_sel[0] + 3 * np.sqrt(sig2_hor_sel[0, 0])])
plt.ylim([0, np.max([np.max(y1[:, 0]), np.max(y1[:, 1])])+10])
plt.xticks()
plt.yticks()
plt.gca().xaxis.set_major_formatter(tck.FuncFormatter(lambda z, _:
                                    '{:.2%}'.format(z)))
l1 = plt.plot(x1[:, 0], y1[:, 0], lw=1.5, color=blue)  # marginal pdf
l2 = plt.plot(x1[:, 1], y1[:, 1], lw=1.5, color=dgrey)  # stationary pdf
l3 = plt.plot([drift_hor_sel[0] - 2*np.sqrt(sig2_hor_sel[0, 0]),
               drift_hor_sel[0] + 2*np.sqrt(sig2_hor_sel[0, 0])], [0.5, 0.5],
              color=orange, lw=2)  # 2 z-score
l4 = plt.plot(x[-1, idx_tau[0]], 0.5, marker='o', color='k',
              markerfacecolor='k', markersize=5)  # initial position
l5 = plt.plot(drift_hor_sel[0], 0.5, color='g', marker='.', markersize=10,
              markerfacecolor='g')  # mean

# marginal and stationary distributions: left plot
ax[1, 0].axis('off')

plt.sca(ax[0, 0])
plt.xlim([0, np.max([np.max(y2[:, 0]), np.max(y2[:, 1])])+10])
plt.ylim([drift_hor_sel[1] - 3 * np.sqrt(sig2_hor_sel[1, 1]),
          drift_hor_sel[1] + 3 * np.sqrt(sig2_hor_sel[1, 1])])
plt.xticks()
plt.yticks()
plt.gca().yaxis.set_major_formatter(tck.FuncFormatter(lambda z, _:
                                    '{:.2%}'.format(z)))
plt.plot(y2[:, 0], x2[:, 0], lw=1.5, color=blue)  # marginal pdf
plt.plot(y2[:, 1], x2[:, 1], lw=1.5, color=dgrey)  # stationary distribution
plt.plot([0.5, 0.5], [drift_hor_sel[1] - 2*np.sqrt(sig2_hor_sel[1, 1]),
                      drift_hor_sel[1] + 2*np.sqrt(sig2_hor_sel[1, 1])],
         color=orange, lw=2)  # 2 z-score
plt.plot(0.5, x[-1, idx_tau[1]], color='k', marker='.', markersize=10,
         markerfacecolor='k')  # initial position
plt.plot(0.5, drift_hor_sel[1], color='g', marker='.', markersize=10,
         markerfacecolor='g')  # mean
leg = plt.legend(handles=[l1[0], l2[0], l3[0], l4[0], l5[0], hs[0]],
                 labels=['Pdf', 'Stationary distribution', '2 z-score',
                         'Current value', 'Mean', 'Horizon scenarios'],
                 bbox_to_anchor=(1, -0.5))

add_logo(fig, axis=ax[1, 1], size_frac_x=1/8, location=1)
# -

t_m.shape
t_m[-1]

t_now


