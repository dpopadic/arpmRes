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

# # s_exp_decay_fp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_exp_decay_fp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerExpDecProbs).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation import effective_num_scenarios, exp_decay_fp
from arpym.tools import colormap_fp, histogram_sp, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_exp_decay_fp-parameters)

tau_hl = 750

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_exp_decay_fp-implementation-step00): Upload data

# S&P 500 index value
spx_path = '../../../databases/global-databases/equities/db_stocks_SP500/SPX.csv'
spx_all = pd.read_csv(spx_path, parse_dates=['date'])
spx = spx_all.loc[(spx_all['date'] >= pd.to_datetime('2004-01-02')) &
                  (spx_all['date'] < pd.to_datetime('2017-09-01'))]

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_exp_decay_fp-implementation-step01): Compute the S&P 500 compounded return

# invariants (S&P500 log-return)
epsi = np.diff(np.log(spx.SPX_close))  # S&P 500 index compounded return

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_exp_decay_fp-implementation-step02): Compute the time exponential decay probabilities

t_ = len(epsi)
t_star = t_
p_exp = exp_decay_fp(t_, tau_hl, t_star)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_exp_decay_fp-implementation-step03): Compute the effective number of scenarios

ens = effective_num_scenarios(p_exp)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_exp_decay_fp-implementation-step04): Compute flexible probabilities histogram

f_eps, x_eps = histogram_sp(epsi, p=p_exp, k_=10*np.log(t_))

# ## Plots

# +
# figure settings
plt.style.use('arpm')
grey_range = np.r_[np.arange(0, 0.6 + 0.01, 0.01), .85]
[color_map, p_colors] = colormap_fp(p_exp, np.min(p_exp), np.max(p_exp),
                                    grey_range, 0, 1, [1, 0])
myFmt = mdates.DateFormatter('%d-%b-%Y')
bar_dates = np.array(spx.date[1:])

# flexible probabilities profile
f, ax = plt.subplots(2, 1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.sca(ax[0])
ax[0].bar(bar_dates, p_exp, color='gray',
          width=np.floor(len(p_exp)/680))
for label in ax[0].xaxis.get_ticklabels():
    label.set_fontsize(14)
ax[0].set_yticks([])
ax[0].set_xlim(min(spx.date[1:]), max(spx.date[1:]))
plt.ylim([np.min(p_exp), np.max(p_exp)])
ax[0].xaxis.set_major_formatter(myFmt)
plt.ylabel('probability', fontsize=17)
txt1 = 'Effective num. scenarios: % 3.0f\n' % ens
txt5 = 'Half-life (days): % 3.0f' % tau_hl
plt.title('Exponential decay probabilities\n'+txt1+txt5,
          fontsize=20, fontweight='bold')

# scatter plot color-coded
plt.sca(ax[1])
plt.xlim(min(spx.date[1:]), max(spx.date[1:]))
plt.ylim(-0.15, 0.15)
plt.scatter(np.array(spx.date[1:]), epsi, s=3, c=p_colors, marker='*',
            cmap=color_map)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax[1].xaxis.set_major_formatter(myFmt)
plt.ylabel(r'invariant $\epsilon_t$', fontsize=17)
plt.title('S&P 500', fontsize=20, fontweight='bold')
add_logo(f, set_fig_size=False)
plt.tight_layout()

# HFP histogram
hfp = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
ax = hfp.add_subplot(111)
plt.xlim(-0.15, 0.15)
bar_width = x_eps[1] - x_eps[0]
ax.bar(x_eps, f_eps, width=bar_width, fc=[0.7, 0.7, 0.7],
       edgecolor=[0.5, 0.5, 0.5])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_title('HFP distribution', fontsize=20, fontweight='bold')
add_logo(hfp, set_fig_size=False)
plt.tight_layout()
