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

# # s_projection_brownian_motion [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_projection_brownian_motion&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerBrowMotProj).

# +
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from arpym.estimation import exp_decay_fp
from arpym.statistics import meancov_sp, simulate_bm
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_projection_brownian_motion-parameters)

t_ = 504  # time series length
j_ = 100  # number of scenarios
delta_t_m = np.array([1, 1, 2, 1, 3, 1, 1])  # time to horizon (days)
tau_hl = 180  # half-life (days)

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_projection_brownian_motion-implementation-step00): Upload data

# +
path = '../../../databases/global-databases/equities/'

# import data
df_stocks = pd.read_csv(path + 'db_stocks_SP500/db_stocks_sp.csv', index_col=0,
                        skiprows=[0])

# set timestamps
df_stocks = df_stocks.set_index(pd.to_datetime(df_stocks.index))

# select data within the date range
df_stocks = df_stocks.loc[df_stocks.index].tail(t_)

# select stock
df_stocks = df_stocks['AMZN']  # stock value
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_projection_brownian_motion-implementation-step01): Compute the risk driver

x = np.log(np.array(df_stocks))  # log-value

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_projection_brownian_motion-implementation-step02): Compute HFP mean and covariance

epsi = np.diff(x)  # invariant past realizations
p = exp_decay_fp(t_ - 1, tau_hl)  # exponential decay probabilities
mu_hat, sig2_hat = meancov_sp(epsi, p)  # HFP mean and covariance

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_projection_brownian_motion-implementation-step03): Compute Monte Carlo paths of risk drivers

x_tnow_thor = simulate_bm(x[-1].reshape(1), delta_t_m, mu_hat.reshape(1),
                          sig2_hat.reshape((1, 1)), j_).squeeze()

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_projection_brownian_motion-implementation-step04): Compute projected expectations and standard deviations

mu_thor = x[-1] + mu_hat * np.cumsum(delta_t_m)  # projected expectations
sig_thor = np.sqrt(sig2_hat) * np.sqrt(np.cumsum(delta_t_m))  # projected standard deviations

# ## Plots

# +
# analytical pdf at horizon
l_ = 2000  # number of points
x_pdf_hor = np.linspace(mu_thor[-1] - 4 * sig_thor[-1],
                        mu_thor[-1] + 4 * sig_thor[-1], l_)
y_pdf_hor = st.norm.pdf(x_pdf_hor, mu_thor[-1], sig_thor[-1])

# preliminary settings
plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.2, 0.2, 0.2]  # dark grey

s_ = 2  # number of plotted observation before projecting time

fig = plt.figure()

# axes settings
m = np.min([np.min(x[-2:]), mu_thor[-1] - 4 * sig_thor[-1]])
M = np.max([np.max(x[-2:]), mu_thor[-1] + 4.5 * sig_thor[-1]])
t1 = np.r_[0, np.cumsum(delta_t_m)]
t = np.concatenate((np.arange(-s_, 0), t1))
max_scale = np.sum(delta_t_m) / 4
scale = max_scale / np.max(y_pdf_hor)
plt.axis([t[0], t[-1] + max_scale, m, M])
plt.xlabel('time (days)')
plt.ylabel('Log-value')
plt.yticks()
plt.grid(False)
plt.title('Projection of Brownian motion')

# simulated paths
plt.plot(t1, x_tnow_thor.T, color=lgrey, lw=2)

# expectation and standard deviation lines
timetohor_t_now = np.sum(delta_t_m)
t_line = np.arange(0, timetohor_t_now + 0.01, 0.01)
mu_line = x[-1] + mu_hat * t_line
sig_line = np.sqrt(sig2_hat * t_line)
p_mu = plt.plot(t_line, mu_line, color='g',
                label='expectation', lw=2)
plt.plot(t_line, mu_line + 2 * sig_line, color='r',
         label='+ / - 2 st.deviation', lw=2)
plt.plot(t_line, mu_line - 2 * sig_line, color='r', lw=2)

# analytical pdf at the horizon plot
for k, y in enumerate(y_pdf_hor):
    plt.plot([timetohor_t_now, timetohor_t_now + y_pdf_hor[k] * scale],
             [x_pdf_hor[k], x_pdf_hor[k]],
             color=lgrey, lw=2)

plt.plot(timetohor_t_now + y_pdf_hor * scale, x_pdf_hor,
         color=dgrey, label='horizon pdf', lw=1)

# plot of last s_ observations
for k in range(s_):
    plt.plot([t[k], t[k + 1]], [x[- s_ + k - 1], x[- s_ + k]],
             color=lgrey, lw=2)
    plt.plot(t[k], x[- s_ + k - 1],
             color='b', linestyle='none', marker='.', markersize=15)

plt.plot(t[s_], x[-1], color='b', linestyle='none', marker='.', markersize=15)

# legend
plt.legend()

add_logo(fig)
plt.tight_layout()
