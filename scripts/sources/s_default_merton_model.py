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

# # s_default_merton_model [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_default_merton_model&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-merton-struct-model).

# +
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-parameters)

v_asset_t = 10  # initial asset value
mu_asset = 0  # "percentage" drift of the GBM
sigma_asset = 0.3  # "percentage" volatility of the GBM
j_ = 5  # number of trajectories for the plot
v_liab_t = 6  # initial value of the liabilities
r = 0.1  # liabilities growth coefficient
n_steps = 252  # number of time steps between t and t+1

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step01): Monte Carlo scenarios for the path of the asset value

delta_tm = 1/n_steps
epsi = (mu_asset - 0.5 * sigma_asset ** 2) * delta_tm +\
       sigma_asset * np.sqrt(delta_tm) * stats.norm.rvs(size=(n_steps, j_))
epsi[0, 0] = 0
v_asset = v_asset_t * np.exp(np.cumsum(epsi, axis=0))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step02): Distribution of the assets value at time t+1

n_grid = 100
lognscale = np.exp(np.log(v_asset_t) + (mu_asset - 0.5 * sigma_asset ** 2))
x_grid = np.linspace(stats.lognorm.ppf(.01, sigma_asset, scale=lognscale),
                     stats.lognorm.ppf(.99, sigma_asset, scale=lognscale),
                     n_grid)
f_vasset_tp1 = stats.lognorm.pdf(x_grid, sigma_asset, scale=lognscale)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step03): Liabilities evolution

t_plot = np.linspace(0, 1, n_steps)
v_liab = v_liab_t * np.exp(t_plot * r)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step04): Log-leverage and probability of default

l_t = np.log(v_liab_t / v_asset_t)
mu_l = -mu_asset + 0.5 * sigma_asset ** 2 + r
sigma_l = sigma_asset
dd_t = (l_t+mu_l)/sigma_l
p_def_l = stats.norm.cdf(dd_t)

# ## Plots

# +
plt.style.use('arpm')

lblue = [0.58, 0.80, 0.87]  # light blue
lgreen = [0.76, 0.84, 0.61]  # light green
lpurple = [0.70, 0.64, 0.78]  # light purple

fig = plt.figure()

# balance sheet at time 0
ax1 = plt.subplot2grid((3, 3), (0, 0))
plt.bar(0, v_asset_t, width=1, facecolor=lblue)
plt.bar(1, np.max([v_asset_t - v_liab_t, 0]), width=1, facecolor=lgreen)
plt.bar(1,  v_liab_t, bottom=[np.max(v_asset_t - v_liab_t, 0)], width=1,
        facecolor=lpurple)
plt.axis([0, 3, 0, np.max(v_asset)])
ax1.xaxis.set_visible(False)
plt.title('Balance sheet at time t')
ax1.legend(['Assets', 'Equities', 'Liabilities'], loc='best')

# balance sheet at time 1
ax3 = plt.subplot2grid((3, 3), (0, 2))
plt.bar(0, v_asset[-1, -1], width=1, facecolor=lblue)
plt.bar(1, np.max([v_asset[-1, -1] - v_liab[-1], 0]), width=1,
        facecolor=lgreen)
plt.bar(1, v_liab[-1], bottom=np.max([v_asset[-1, -1] - v_liab[-1], 0]),
        width=1, facecolor=lpurple)
plt.axis([0, 3, 0., np.max(v_asset)])
ax3.xaxis.set_visible(False)
plt.title('Balance sheet at time t+1')
ax3.legend(['Assets', 'Equities', 'Liabilities'], loc='best')

# probability of default at time 1
ax2 = plt.subplot2grid((3, 3), (0, 1))
plt.plot(l_t, p_def_l, 'r.')
plt.ylim([0, 1])
plt.title('Probability of default at time t+1')
plt.xlabel('Log-leverage')
plt.ylabel('Probability')

ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2)
plt.xticks([0, 1], ['t', 't+1'])

# liabilities
plt.plot(t_plot, v_liab, color=lpurple, lw=1)

# assets
plt.plot(t_plot, v_asset[:, -1], color=lblue, lw=2)
for j in range(j_):
    plt.plot(t_plot, v_asset[:, j], color=lblue, lw=0.75)
plt.plot(0, v_asset_t, '.', color=lblue, markersize=20)

# assets distribution at time 1
pdf_border = plt.plot(1+f_vasset_tp1, x_grid, color=lblue, lw=1)
idx_def = np.where(x_grid <= v_liab[-1])[0]

# highlight solvent vs default areas under the pdf
ax4.fill_betweenx(x_grid, np.ones(n_grid), 1+f_vasset_tp1, facecolor=lblue)
ax4.fill_betweenx(x_grid[idx_def], np.ones(len(idx_def)), 1+f_vasset_tp1[idx_def],
                  facecolor='r')
pdf_border_inf = plt.plot(np.ones(n_grid), x_grid, color='k')
plt.legend(['Liabilities', 'Assets'])

add_logo(fig)
plt.tight_layout()
