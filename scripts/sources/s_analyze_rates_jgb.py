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

# # s_analyze_rates_jgb [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_analyze_rates_jgb&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-log-shad-rates-risk-driv).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.pricing import ytm_shadowrates
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_analyze_rates_jgb-parameters)

tau_select = [1, 2, 3, 5, 7, 10, 20]  # selected times to maturity (years)
eta = 0.013  # smoothing parameter for call function

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_analyze_rates_jgb-implementation-step00): Upload data

path = '../../../databases/global-databases/fixed-income/db_japanesegovbond/'
df_data = pd.read_csv(path + 'data.csv',
                             header=0,
                             index_col=0,
                             parse_dates=['date'],
                             infer_datetime_format=True)
tau = pd.read_csv(path + 'params.csv').iloc[:, 0].values

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_analyze_rates_jgb-implementation-step01): Select yields

# +
tau_select = np.sort(tau_select)
tau_index = np.searchsorted(tau, tau_select)

t = df_data.index.values
y = df_data.iloc[:, tau_index].values  # yields for selected times to maturity
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_analyze_rates_jgb-implementation-step02): log-yields and shadow rates

log_y = np.log(y)  # log-yields for selected times to maturity
c_inv_eta = ytm_shadowrates(y, eta)  # shadow rates for selected times to maturity

# ## Plots

# +
plt.style.use('arpm')
cmap = plt.get_cmap('Reds_r')
colors = [cmap(i) for i in np.linspace(0, 0.8, tau_select.shape[0])]

time_to_maturity = [str(x) + ' year' if x == 1
                    else str(x) + ' years'
                    for x in tau_select]

myFmt = mdates.DateFormatter('%d-%b-%y')
xtick_count = 6
fig, ax = plt.subplots(3, 1, sharex=True)

handles = []
for yields, log_yields, shadow_rates, c in zip(y.T, log_y.T, c_inv_eta.T, colors):
    ax[0].plot(t, yields, c=c, lw=0.5)
    ax[1].plot(t, log_yields, c=c, lw=0.5)
    line, = ax[2].plot(t, shadow_rates, c=c, lw=0.5)
    handles.append(line)

ax[0].set_ylabel('Yields')
ax[1].set_ylabel('Log-yields')
ax[2].set_ylabel('Shadow rates')
ax[2].xaxis.set_ticks(t[np.linspace(0, len(t)-1, xtick_count, dtype=int)])
ax[2].xaxis.set_major_formatter(myFmt)

ax[0].set_ylim(np.min(y), np.max(y))
ax[1].set_ylim(np.min(log_y), np.max(log_y))
ax[2].set_ylim(np.min(c_inv_eta), np.max(c_inv_eta))
ax[2].set_xlim(np.min(t), np.max(t))

fig.suptitle('Japanese government bond yields',
             x=0.5, y=1,
             fontweight='semibold')

fig.legend(handles,
           time_to_maturity,
           loc='center',
           ncol=len(time_to_maturity),
           bbox_to_anchor=(0.5, 0.01),
           columnspacing=0.25,
           handletextpad=0.1)

add_logo(fig)
plt.tight_layout()
