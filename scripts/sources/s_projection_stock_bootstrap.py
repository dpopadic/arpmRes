#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# # s_projection_stock_bootstrap [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_projection_stock_bootstrap&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_bootstrap).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.estimation import exp_decay_fp
from arpym.statistics import simulate_rw_hfp, meancov_sp
from arpym.tools import histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_bootstrap-parameters)

# +
stock = 'AMZN'  # S&P 500 company (ticker)
t_ = 504  # length of the stock value time series
tau_hl = 180  # half life (days)
m_ = 10  # number of monitoring times
j_ = 1000  # number of scenarios
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_bootstrap-implementation-step00): Upload data

# +
path = '../../../databases/global-databases/equities/db_stocks_SP500/'
df_stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0], index_col=0)

# select data
df_stocks = df_stocks[stock].tail(t_)

# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_bootstrap-implementation-step01): Compute risk driver

# +
x = np.log(np.array(df_stocks))  # log-value
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_bootstrap-implementation-step02): HFP distribution of the invariant

# +
epsi = np.diff(x)  # historical scenarios
p = exp_decay_fp(t_ - 1, tau_hl)  # probabilities
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_bootstrap-implementation-step03): Generate scenarios of log-value via bootstrapping

# +
x_tnow_thor = simulate_rw_hfp(x[-1].reshape(1), epsi, p, j_, m_).squeeze()
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_bootstrap-implementation-step04): Evolution of expectation and standard deviation

# +
mu_thor = np.zeros(m_ + 1)
sig_thor = np.zeros(m_ + 1)
for m in range(0, m_ + 1):
    mu_thor[m], sig2_thor = meancov_sp(x_tnow_thor[:, m].reshape(-1, 1))
    sig_thor[m] = np.sqrt(sig2_thor)
# -

# ## Plots

# +
# preliminary settings
plt.style.use('arpm')
mydpi = 72.0
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.2, 0.2, 0.2]  # dark grey
t_m = np.arange(0, m_ + 1)
j_plot = 40  # number of paths to be plotted
h, b = histogram_sp(x_tnow_thor[:, -1], k_=10 * np.log(j_))
fig, ax = plt.subplots()
ax.set_facecolor('white')
# axis settings
min_x = np.min([np.min(x_tnow_thor[:, :]) - 0.1,
                mu_thor[-1] - 4 * sig_thor[-1]])
max_x = np.max([np.max(x_tnow_thor[:, -1]) + 0.1,
                mu_thor[-1] + 4 * sig_thor[-1]])
plt.axis([t_m[0], t_m[-1] + np.max(h) * 0.2 + 0.03, min_x, max_x])
plt.xlabel('time (days)')
plt.ylabel('Log-value')
plt.xticks(t_m)
plt.yticks()
plt.grid(False)
plt.title('Projection of %s log-value' % (stock))

# simulated paths
plt.plot(t_m.reshape(-1, 1), x_tnow_thor[:j_plot, :].T, color=lgrey, lw=0.5)
p_mu = plt.plot(t_m, mu_thor, color='g', label='expectation', lw=1)
p_red_1 = plt.plot(t_m, mu_thor + 2 * sig_thor, label='+ / - 2 st.deviation',
                   color='r', lw=1)
p_red_2 = plt.plot(t_m, mu_thor - 2 * sig_thor, color='r', lw=1)

# histogram at horizon
h = h * 0.2  # adapt the hist height to the current xaxis scale
emp_pdf = plt.barh(b, h, left=t_m[-1],
                   height=b[1] - b[0], facecolor=lgrey,
                   edgecolor=lgrey, label='horizon pdf')
pdf_border = plt.plot(h + t_m[-1], b, color=dgrey, lw=1)
plt.plot([t_m[-1], t_m[-1]], [b[0], b[-1]], color=dgrey, lw=0.5)

# legend
plt.legend()

add_logo(fig)
plt.tight_layout()
