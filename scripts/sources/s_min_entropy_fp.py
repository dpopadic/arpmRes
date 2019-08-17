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

# # s_min_entropy_fp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_min_entropy_fp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerFPentrpool).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation import conditional_fp, effective_num_scenarios, exp_decay_fp
from arpym.statistics import scoring, smoothing
from arpym.tools import colormap_fp, histogram_sp, add_logo

np.seterr(invalid='ignore')
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_min_entropy_fp-parameters)

z_star = -0.27  # target value
alpha = 0.25  # leeway
tau_hl_prior = 6*252  # prior half life

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_min_entropy_fp-implementation-step00): Upload data

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
t_ = len(date)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_min_entropy_fp-implementation-step01): Compute the S&P 500 compounded return and the VIX compounded return

epsi = np.diff(np.log(spx_vix.SPX_close))  # S&P 500 index compounded return
v_vix = np.array(spx_vix.VIX_close)  # VIX index value
c = np.diff(np.log(v_vix))  # VIX index compounded return

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_min_entropy_fp-implementation-step02): Compute the risk factor by smoothing and scoring VIX compounded return

tau_hl_smoo = 22
tau_hl_scor = 144
z_smooth = smoothing(c, tau_hl_smoo)  # smoothing
z = scoring(z_smooth, tau_hl_scor)  # scoring

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_min_entropy_fp-implementation-step03): Compute the flexible probabilities conditioned via minimum relative entropy

prior = exp_decay_fp(t_-1, tau_hl_prior)
# minimum relative entropy flexible probabilities
p_entropy = conditional_fp(z, z_star, alpha, prior)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_min_entropy_fp-implementation-step04): Compute the effective number of scenarios

ens = effective_num_scenarios(p_entropy)  # effective number of scenarios

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_min_entropy_fp-implementation-step05): Compute the flexible probabilities histogram

n_bins = 10 * np.log(t_-1)
f_eps, x_eps = histogram_sp(epsi, p=p_entropy, k_=n_bins)  # flex. prob. hist.

# ## Plots

# +
plt.style.use('arpm')

grey_range = np.r_[np.arange(0, 0.6 + 0.01, 0.01), .85]
[color_map, p_colors] = colormap_fp(p_entropy, np.min(p_entropy),
                                    np.max(p_entropy), grey_range, 0, 1,
                                    [1, 0])
plot_dates = np.array(date)
myFmt = mdates.DateFormatter('%d-%b-%Y')
date_tick = np.arange(84, t_-2, 800)

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
                      color='C4', lw=1.5)

lns = vixPlot + zPlot + targetPlot
labs = ['VIX', 'Market state',
        'Conditioning state={z_star:.2f}'.format(z_star=z_star)]

ax.legend(lns, labs, loc=2, fontsize=17)

ax.set_xlabel('date', fontsize=17)
ax.set_xlim(min(plot_dates), max(plot_dates))
ax.xaxis.set_major_formatter(myFmt)
ax.set_title('VIX and market state', fontweight='bold', fontsize=20)

ax.grid(False)
ax2.grid(False)
add_logo(fig1, location=1, set_fig_size=False)
plt.tight_layout()

# flexible probabilities plot
fig2, axs = plt.subplots(2, 1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
axs[0].bar(plot_dates[1:], p_entropy, color='gray',
           width=np.floor(len(p_entropy)/680))
for label in axs[0].xaxis.get_ticklabels():
    label.set_fontsize(14)
axs[0].set_yticks([])
axs[0].set_xlim(min(plot_dates), max(plot_dates))
axs[0].xaxis.set_major_formatter(myFmt)
axs[0].set_ylim(0, np.max(p_entropy)*(1+1./7.))
axs[0].set_ylabel('probability', fontsize=17)
axs[0].set_title('State and time conditioning probabilities',
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
ax.set_title('HFP distribution', fontweight='bold', fontsize=20)
add_logo(hfp, set_fig_size=False)
plt.tight_layout()
