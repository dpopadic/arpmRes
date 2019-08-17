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

# # s_state_crisp_fp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_state_crisp_fp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerCrispProb).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation import effective_num_scenarios, crisp_fp
from arpym.statistics import scoring, smoothing
from arpym.tools import colormap_fp, histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_state_crisp_fp-parameters)

z_star = 0.5  # target value
alpha = 0.25  # total probability to be contained in the range

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_state_crisp_fp-implementation-step00): Upload data

spx_path = '../../../databases/global-databases/equities/db_stocks_SP500/SPX.csv'
vix_path = '../../../databases/global-databases/derivatives/db_vix/data.csv'
# S&P 500 index value
spx = pd.read_csv(spx_path, parse_dates=['date'])
# VIX index value
vix = pd.read_csv(vix_path, usecols=['date', 'VIX_close'],
                  parse_dates=['date'])
# merging datasets
spx_vix = pd.merge(spx, vix, how='inner', on=['date'])
date = spx_vix.date

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_state_crisp_fp-implementation-step01): Compute the S&P 500 compounded return and the VIX compounded return

epsi = np.diff(np.log(spx_vix.SPX_close))  # S&P 500 index compounded return
v_vix = np.array(spx_vix.VIX_close)  # VIX index value
c = np.diff(np.log(v_vix))  # VIX index compounded return
t_ = len(epsi)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_state_crisp_fp-implementation-step02): Compute the risk factor by smoothing and scoring VIX compounded return

tau_hl_smoo = 15  # smoothing half-life parameter
tau_hl_scor = 100  # scoring half-life parameter
z_smooth = smoothing(c, tau_hl_smoo)  # smoothing
z = scoring(z_smooth, tau_hl_scor)  # scoring

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_state_crisp_fp-implementation-step03): Compute the state crisp probabilities

p_crisp, z_lb, z_ub = crisp_fp(z, z_star, alpha)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_state_crisp_fp-implementation-step04): Compute the effective number of scenarios

ens = effective_num_scenarios(p_crisp)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_state_crisp_fp-implementation-step05): Compute flexible probabilities histogram

f_eps, x_eps = histogram_sp(epsi, p=p_crisp, k_=10*np.log(t_))

# ## Plots

# +
plt.style.use('arpm')

grey_range = np.r_[np.arange(0, 0.6 + 0.01, 0.01), .85]
[color_map, p_colors] = colormap_fp(p_crisp, np.min(p_crisp),
                                    np.max(p_crisp), grey_range, 0, 1,
                                    [1, 0])
plot_dates = np.array(date)
myFmt = mdates.DateFormatter('%d-%b-%Y')
date_tick = np.arange(84, t_-1, 800)

# VIX and market state
fig1 = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

ax = fig1.add_subplot(111)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
vixPlot = ax.plot(plot_dates, v_vix, color='C3')
ax.set_ylabel('VIX', color='C3', fontsize=17)
ax.tick_params(axis='y', colors='C3')

ax2 = ax.twinx()
plt.yticks(fontsize=14)
zPlot = ax2.plot(plot_dates[1:], z, color='C0', lw=1.15)
ax2.set_ylabel('Market state', color='C0', fontsize=17)
ax2.tick_params(axis='y', colors='C0')
targetPlot = ax2.plot(plot_dates, z_star * np.ones(len(plot_dates)),
                      color='C4', linestyle='--', lw=1.5)
lb = ax2.plot(plot_dates, z_lb * np.ones(len(plot_dates)),
              color='C5', lw=1.5)
ub = ax2.plot(plot_dates, z_ub * np.ones(len(plot_dates)),
              color='C5', linestyle='--', lw=1.5)

lns = vixPlot + zPlot + targetPlot + lb
labs = ['VIX', 'Market state',
        'Conditioning state={z_star:.2f}'.format(z_star=z_star),
        'Conditioning state bounds']

ax.legend(lns, labs, loc=2, fontsize=17)

ax.set_xlabel('date', fontsize=17)
ax.set_xlim(min(plot_dates), max(plot_dates))
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('VIX and market state', fontsize=20, fontweight='bold')

ax.grid(False)
ax2.grid(False)
add_logo(fig1, location=1, set_fig_size=False)
plt.tight_layout()

# state crisp probabilities plot
fig2, axs = plt.subplots(2, 1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
axs[0].bar(plot_dates[1:], p_crisp, color='gray',
           width=np.floor(len(p_crisp)/680))
for label in axs[0].xaxis.get_ticklabels():
    label.set_fontsize(14)
axs[0].set_yticks([])
axs[0].set_xlim(min(plot_dates), max(plot_dates))
axs[0].xaxis.set_major_formatter(myFmt)
axs[0].set_ylim(0, np.max(p_crisp)*(1+1./7.))
axs[0].set_ylabel('probability', fontsize=17)
axs[0].set_title('State crisp probabilities',
                 fontweight='bold', fontsize=20)
plt.sca(axs[1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
axs[1].set_xlim(min(plot_dates), max(plot_dates))
axs[1].set_ylim(-0.15, 0.15)
axs[1].scatter(plot_dates[1:], epsi, s=30, c=p_colors, marker='.',
               cmap=color_map)
axs[1].set_facecolor("white")
axs[1].set_title('S&P 500', fontweight='bold', fontsize=20)
axs[1].set_ylabel('return', fontsize=17)
axs[1].xaxis.set_major_formatter(myFmt)
add_logo(fig2, axis=axs[1], location=1, set_fig_size=False)
plt.tight_layout()

# HFP histogram
hfp = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
ax = hfp.add_subplot(111)
bar_width = x_eps[1] - x_eps[0]
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_xlim(-0.15, 0.15)
ax.bar(x_eps, f_eps, width=bar_width, fc=[0.7, 0.7, 0.7],
       edgecolor=[0.5, 0.5, 0.5])
ax.set_title('HFP distribution', fontweight='bold', fontsize=17)
add_logo(hfp, set_fig_size=False)
plt.tight_layout()
