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

# # s_simulate_call [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_simulate_call&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-simeucall).

# +
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from arpym.pricing import bsm_function
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-parameters)

# +
j_ = 30  # number of scenarios (=number of basis call options)
tau_hor = 60  # time to horizon
s_low = 77.66  # lower bound for the underlying grid
gamma = 2.9  # tick-size of underlying/strikes at expiry
s_tnow = 120  # underlying current value
r = 2 * 1e-4  # risk-free interest rate
sigma = 0.01  # volatility of the underlying
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step01): Underlying scenarios at horizon and striks' calls

# +
k_strk = (s_low + np.arange(0, j_, 1).reshape(1, -1)*gamma).T
s_thor = (k_strk + gamma).T
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step02): Payoff matrix of basis call options

# +
v_basis = np.triu(toeplitz(np.arange(1, j_+1)))*gamma
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step03): Current values basis call options

# +
v_call_tnow = np.zeros(j_)
for n in range(j_):
    m = np.log(s_tnow/k_strk[n])/np.sqrt(tau_hor)
    v_call_tnow[n] = bsm_function(s_tnow, r, sigma, m, tau_hor)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step04): Current values AD securites

# +
v_ad_tnow = np.linalg.solve(v_basis, v_call_tnow)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step05): Save databases

# +
out = np.c_[s_thor.reshape(-1,1), v_ad_tnow]
col = ['s_thor', 'v_ad_tnow']
out = pd.DataFrame(out, columns=col)
out.to_csv('../../../databases/temporary-databases/db_simcall.csv')
del out
out = {'s_tnow': pd.Series(s_tnow),
       'tau_hor': pd.Series(tau_hor),
       'gamma': pd.Series(gamma),
       'r': pd.Series(r),
       'sigma_tnow': pd.Series(sigma)}
out = pd.DataFrame(out)
out.to_csv('../../../databases/temporary-databases/db_simcall_tools.csv')
del out
# -

# ## Plots

# +
s_up = s_thor[0, -1]  # upper bound for the underlying/strike grid
plt.style.use('arpm')
fig = plt.figure()


ax = plt.subplot2grid((8, 11), (7, 0), colspan=7)
ax.imshow(100*v_ad_tnow.reshape(1, -1), vmin=0, vmax=s_up - s_low, cmap=cm.jet, aspect='auto')
plt.xticks([])
plt.yticks([])
plt.title(r'$100*v^{\mathit{AD}}_{t_{\mathit{now}}}$')
plt.grid(False)


plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1.25
ax1 = plt.subplot2grid((4, 11), (0, 0), colspan=7, rowspan=3)
ax1.imshow(v_basis, cmap=cm.jet, aspect='auto')
plt.xlabel('scenario')
plt.ylabel('instrument')
plt.title(r'$V_{t_{\mathit{now}}\rightarrow t_{\mathit{hor}}}^{\mathit{payoff}}$')
plt.yticks(np.arange(4, 30, 5), np.arange(5, 31, 5))
plt.xticks(np.arange(4, 30, 5), np.arange(5, 31, 5))
plt.grid(False)

ax = plt.subplot2grid((4, 11), (0, 8), rowspan=3)
ax.imshow(v_call_tnow.reshape(-1, 1), vmin=0, vmax=s_up - s_low, cmap=cm.jet, aspect='auto')
plt.xticks([])
plt.yticks(np.arange(4, 30, 5), np.arange(5, 31, 5))
plt.title(r'$v^{\mathit{call}}_{t_{\mathit{now}}}$')
plt.grid(False)

ax = plt.subplot2grid((4, 11), (0, 10), rowspan=3)
cbar = np.floor((np.flipud(s_thor.T - s_low)) * 100) / 100
plt.imshow(cbar, cmap=cm.jet, aspect='auto')
plt.xticks([])
plt.yticks([0, 7, 15, 23, 29], [i[0] for i in cbar[[0, 7, 15, 23, 29]]])
plt.title('Scale')
plt.grid(False)

add_logo(fig, axis=ax1, location=3, size_frac_x=1/12)
