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

# # s_aggregation_options_hfp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_aggregation_options_hfp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-aggr-hfp).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.statistics import meancov_sp
from arpym.tools import colormap_fp
from arpym.tools import histogram_sp
from arpym.tools.logo import add_logo

from arpym.estimation import effective_num_scenarios
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_options_hfp-parameters)

h = np.array([1, 1])  # holdings

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_options_hfp-implementation-step00): Extract data from db_pric_options

# +
path = '../../../databases/temporary-databases/'
# read the database
df = pd.read_csv(path + 'db_pric_options.csv', index_col=0)

pi_call = np.array(df['pi_call'])  # call option P&L scenarios
pi_put = np.array(df['pi_put'])  # put option P&L scenarios
p = np.array(df['p'])  # probabilities
dates = np.array(df.index.values)  # dates
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_options_hfp-implementation-step01): Compute the portfolio P&L scenarios and distribution

# +
pi_h = h.T@np.r_[[pi_call], [pi_put]]  # portfolio P&L scenarios
ens = effective_num_scenarios(p)  # effective number scenarios

# mean and standard deviation of the portfolio P&L distribution
[mu_pi_h, sigma2_pi_h] = meancov_sp(pi_h, p)
sigma_pi_h = np.sqrt(sigma2_pi_h)

# mean and standard deviation of the call option P&L distribution
[mu_pi_call, sigma2_pi_call] = meancov_sp(pi_call, p)
sigma_pi_call = np.sqrt(sigma2_pi_call)

# mean and standard deviation of the put option P&L distribution
[mu_pi_put, sigma2_pi_put] = meancov_sp(pi_put, p)
sigma_pi_put = np.sqrt(sigma2_pi_put)
# -

# ## Plots

# +
plt.style.use('arpm')

colhist = [.9, .9, .9]
colhistedge = [.4, .4, .4]
len_pi1 = len(pi_call)
len_pi2 = len(pi_put)
len_pih = len(pi_h)

d = np.linspace(0, len_pi1-1, 4, dtype='int')

colors = np.tile(np.arange(0, 0.85, 0.05), (3, 1)).T
cm, fpcolors = colormap_fp(p, np.min(p), np.max(p), colors)

myFmt = mdates.DateFormatter('%d-%b-%y')

# call option P&L
fig, ax = plt.subplots(2, 1)
# scatter plot
dates = pd.to_datetime(dates)
ax[0].scatter(dates, pi_call, c=fpcolors, marker='.', cmap=cm)
ax[0].axis([min(dates), max(dates), np.min(pi_call), np.max(pi_call)])
ax[0].set_xticks(dates[d])
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_title('Scatter plot call P&L')

# histogram
n_bins = np.round(20 * np.log(ens))
height_1, center_1 = histogram_sp(pi_call, p=p, k_=n_bins)
ax[1].bar(center_1, height_1, facecolor=colhist, edgecolor=colhistedge)
ax[1].set_xlim([np.min(pi_call), np.max(pi_call)])
ax[1].set_title('Histogram call P&L')
s1 = 'Mean   %1.3e \nSdev    %1.3e ' % (mu_pi_call, sigma_pi_call)

plt.text(np.max(pi_call), np.max(height_1), s1, horizontalalignment='right',
         verticalalignment='top')
add_logo(fig)
plt.tight_layout()

# put option P&L
fig, ax = plt.subplots(2, 1)
# scatterplot
ax[0].scatter(dates, pi_put, c=fpcolors, marker='.', cmap=cm)
ax[0].axis([min(dates), max(dates), np.min(pi_put), np.max(pi_put)])
ax[0].set_xticks(dates[d])
myFmt = mdates.DateFormatter('%d-%b-%y')
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_title('Scatter plot put P&L')

# histogram
n_bins = np.round(20 * np.log(ens))
height_2, center_2 = histogram_sp(pi_call, p=p, k_=n_bins)
ax[1].bar(center_2, height_2, facecolor=colhist, edgecolor=colhistedge)
ax[1].set_xlim([np.min(pi_put), np.max(pi_put)])
ax[1].set_title('Histogram put P&L')
s2 = 'Mean   %1.3e \nSdev    %1.3e ' % (mu_pi_put, sigma_pi_put)

plt.text(np.max(pi_put), np.max(height_2), s2, horizontalalignment='right',
         verticalalignment='top')
add_logo(fig)
plt.tight_layout()

# portfolio P&L (long call option + short put option)
fig, ax = plt.subplots(2, 1)
# scatter plot
ax[0].scatter(dates, pi_h, c=fpcolors, marker='.', cmap=cm)
ax[0].axis([min(dates), max(dates), np.min(pi_h), np.max(pi_h)])
ax[0].set_xticks(dates[d])
myFmt = mdates.DateFormatter('%d-%b-%y')
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].set_title('Scatter plot portfolio P&L')

# histogram
n_bins = np.round(20 * np.log(ens))
height_h, center_h = histogram_sp(pi_h, p=p, k_=n_bins)

ax[1].bar(center_h, height_h, facecolor=colhist, edgecolor=colhistedge)
ax[1].set_xlim([np.min(pi_h), np.max(pi_h)])
ax[1].set_title('Histogram portfolio P&L')
sh = 'Mean   %1.3e \nSdev    %1.3e ' % (mu_pi_h, sigma_pi_h)

plt.text(np.max(pi_h), np.max(height_h), sh, horizontalalignment='right',
         verticalalignment='top')

add_logo(fig)
plt.tight_layout()
