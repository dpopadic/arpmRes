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

# # s_simulate_payoff [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_simulate_payoff&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-simulate-payoff).

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from arpym.statistics import simulate_normal
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_payoff-parameters)

n_ = 50  # number of instruments
j_ = 50  # number of scenarios
rf = 0.05  # risk-free rate
a_p, b_p = 0.7, 1  # window for non-normalized probabilities of the scenarios
a_mu, b_mu = -0.3, 0.7  # window for random shifts of the payoffs
a_sd, b_sd = 0.8, 1  # window for random rescales of the payoffs
rho = 0.7  # correlation between initial normal variables that are used to generate the payoffs

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_payoff-implementation-step01): Generate the normal vector

c2 = (1 - rho)*np.eye(n_) + rho*np.ones((n_, n_))  # correlation matrix
x = simulate_normal(np.zeros(n_), c2, j_**2)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_payoff-implementation-step02): Generate the payoff matrix at time u = t + 2

# +
mu = np.exp(0.5)  # expectation of std lognormal distribution
sd = mu * np.sqrt(np.exp(1) - 1)  # stdev of std lognormal distribution

v_t2 = np.ones((j_**2, n_))
v_t2[:, 1] = np.exp(x[:, 1]) / sd
v_t2[:, 2::2] = (np.exp(x[:, 2::2])-mu)/sd
v_t2[:, 3::2] = (-np.exp(-x[:, 3::2])+mu)/sd
v_t2[:, 2:] = v_t2[:, 2:] * np.random.uniform(a_sd, b_sd, n_ - 2)  # scale
v_t2[:, 2:] = v_t2[:, 2:] + np.random.uniform(a_mu, b_mu, n_ - 2)  # shift
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_payoff-implementation-step03): Compute probabilities at time p_t2, p_t1 and conditional probabilities p_t1t2

p_t2 = np.random.uniform(a_p, b_p, j_**2)
p_t2 = p_t2 / np.sum(p_t2)
p_t1 = np.sum(p_t2.reshape((j_, j_)), axis=0)
p_t1t2 = p_t2 / np.reshape(np.tile(p_t1, (j_, 1)).T, j_**2)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_payoff-implementation-step04): Simulate conditional SDF and values V_t1 at time t + 1

# +
sdf_t1t2 = np.random.rand(j_**2) * 0.01
v_t1 = np.zeros((j_, n_))

for j in range(j_):
    ind = j_ * (j - 1) + np.arange(j_)
    sdf_t1t2[ind] = sdf_t1t2[ind] / (v_t2[ind, 0] @ (p_t1t2[ind] *
                                                     sdf_t1t2[ind]))
    v_t1[j, :] = (p_t1t2[ind] * sdf_t1t2[ind]) @ v_t2[ind, :]
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_payoff-implementation-step05): Simulate SDF_t1 and current values (at time t)

sdf_t1 = np.random.rand(j_) * 0.01
sdf_t1 = sdf_t1 / (v_t1[:, 0] @ (p_t1 * sdf_t1)) / (1 + rf)
v_t = (sdf_t1 * p_t1) @ v_t1
ind = np.argsort(v_t)[::-1]

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_payoff-implementation-step06): Compute SDF_t2 and risk neutral densities

sdf_t2 = np.reshape(np.tile(sdf_t1, (j_, 1)).T, j_**2) * sdf_t1t2
q_t2 = p_t2 * sdf_t2 * v_t2[:, 0] / v_t[0]
q_t1 = p_t1 * sdf_t1 * v_t1[:, 0] / v_t[0]

# ## Plots

# +
v_max = 1.20  # upper bound for the values to be displayed
v_min = -0.80  # lower bound for the values to be displayed

v_t1_trunc = v_t1.copy()
v_t1_trunc[v_t1_trunc < v_min] = v_min
v_t1_trunc[v_t1_trunc > v_max] = v_max

v_t2_trunc = v_t2.copy()
v_t2_trunc[v_t2_trunc < v_min] = v_min
v_t2_trunc[v_t2_trunc > v_max] = v_max

plt.style.use('arpm')

# Heatmap of V_t1
fig1 = plt.figure()
ax = plt.subplot2grid((1, 11), (0, 0), colspan=7)
ax.imshow(v_t1_trunc[:, ind].T, cmap=cm.jet, aspect='auto')
plt.xlabel('scenario')
plt.ylabel('instrument')
plt.title('Payoff at $t+1$')
plt.grid(False)

ax = plt.subplot2grid((1, 11), (0, 8))
ax.imshow(v_t[ind].reshape(-1, 1), cmap=cm.jet, aspect='auto')
plt.xticks([])
plt.ylabel('instrument')
plt.title('Current value')
plt.grid(False)

ax = plt.subplot2grid((1, 11), (0, 10))
cbar = np.linspace(v_max, v_min, 200)
plt.imshow(cbar.reshape(-1, 1), cmap=cm.jet, aspect='auto')
plt.xticks([])
tick = np.linspace(0, 199, 10, dtype=int)
plt.yticks(tick, np.round(cbar[tick], decimals=1))
plt.title('Scale')
plt.grid(False)
add_logo(fig1, axis=ax, size_frac_x=3/4)

# Heatmap of V_t2
fig2 = plt.figure()
ax = plt.subplot2grid((1, 11), (0, 0), colspan=7)
ax.imshow(v_t2_trunc[:, ind].T, cmap=cm.jet, aspect='auto')
plt.xlabel('scenario')
plt.ylabel('instrument')
plt.title('Payoff at $t+2$')
plt.grid(False)

ax = plt.subplot2grid((1, 11), (0, 8))
ax.imshow(v_t[ind].reshape(-1, 1), cmap=cm.jet, aspect='auto')
plt.xticks([])
plt.ylabel('instrument')
plt.title('Current value')
plt.grid(False)

ax = plt.subplot2grid((1, 11), (0, 10))
cbar = np.linspace(v_max, v_min, 200)
plt.imshow(cbar.reshape(-1, 1), cmap=cm.jet, aspect='auto')
plt.xticks([])
tick = np.linspace(0, 199, 10, dtype=int)
plt.yticks(tick, np.round(cbar[tick], decimals=1))
plt.title('Scale')
plt.grid(False)
add_logo(fig2, axis=ax, size_frac_x=3/4)

# Histograms
n_plot = 3
fig3 = plt.figure()
for i in range(n_plot):
    f_t2, x_t2 = histogram_sp(v_t2[:, 2*i+1] / v_t2[:, 0], p=q_t2, k_=100)
    f_t1, x_t1 = histogram_sp(v_t1[:, 2*i+1] / v_t1[:, 0], p=q_t1, k_=20)
    x_m = v_t[2*i+1] / v_t[0]

    ax = plt.subplot2grid((n_plot, 2), (i, 0))
    plt.barh(x_t2, f_t2, x_t2[1]-x_t2[0],
             color=[0.8, 0.8, 0.8], edgecolor=[0, 0, 0])
    plt.plot([0, np.max(f_t2)+0.01], [x_m, x_m], 'r')
    plt.ylim([x_m-3, x_m+2])

    ax = plt.subplot2grid((n_plot, 2), (i, 1))
    plt.barh(x_t1, f_t1, x_t1[1]-x_t1[0],
             color=[0.8, 0.8, 0.8], edgecolor=[0, 0, 0])
    plt.plot([0, np.max(f_t1)+0.01], [x_m, x_m], 'r')
    plt.ylim([x_m-3, x_m+2])

add_logo(fig3, axis=ax, size_frac_x=1/8)
plt.tight_layout()
