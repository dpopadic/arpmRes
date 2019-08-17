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

# # s_fit_yield_ns [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fit_yield_ns&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerYieldNelSig).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.pricing import fit_nelson_siegel_yield, nelson_siegel_yield
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yield_ns-parameters)

# +
t_ = 450  # len of the selected time series to plot
tau_select = np.array(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0',
                       '7.0', '8.0', '9.0', '10.0', '15.0', '20.0'])  # times to maturity
theta_init=0.5*np.ones(4)
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yield_ns-implementation-step00): Load data

# +
path = '../../../databases/global-databases/fixed-income/db_yields/'
df_data = pd.read_csv(path + 'data.csv', header=0, index_col=0,
                             parse_dates=True, infer_datetime_format=True)
df_data = df_data.iloc[-t_:]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yield_ns-implementation-step01): Yields for the selected times to maturity

# +
df_data = df_data.loc[:, tau_select]
tau_select = np.array([float(tau_select[i]) for i in range(len(tau_select))])
y = df_data.values
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yield_ns-implementation-step02): Realized Nelson-Siegel model parameters and fitted yield curve

# +
theta = np.zeros((t_, 4))
y_ns = np.zeros((t_, len(tau_select)))

for t in range(t_):
    if t==0:
        theta[t] = fit_nelson_siegel_yield(tau_select, y[t], theta_0=theta_init)
    else:
        theta[t] = fit_nelson_siegel_yield(tau_select, y[t], theta_0=theta[t-1])
    y_ns[t, :] = nelson_siegel_yield(tau_select, theta[t])
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_fit_yield_ns-implementation-step03): Save databases

# +
out = pd.DataFrame(theta)
out.to_csv('../../../databases/temporary-databases/db_fit_yield_ns.csv',
               index=None)
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure()
plt.plot(tau_select, y_ns[t_-1, :], 'b', tau_select, y[t_-1, :], 'r.')
plt.axis([np.min(tau_select), np.max(tau_select), np.min(y_ns[t_-1, :]), np.max(y_ns[t_-1, :])])
plt.xlabel('Time to Maturity')
plt.ylabel('Rate')
plt.legend(['Fit','Rates'])
plt.grid(True)
add_logo(fig)
plt.tight_layout()

f,ax = plt.subplots(4,1, sharex=True)
cellc = ['m','b','g','r']
celll = ['Level','Slope','Curvature','Decay']

t = np.array(df_data.index)

for i in range(3):
    plt.sca(ax[i])
    plt.plot_date(t, theta[:, i], color = cellc[i], fmt='-')
    plt.ylabel(celll[i])
    plt.grid(True)
    plt.xticks([])
    plt.xlim([t[0], t[-1]])

plt.sca(ax[3])
plt.plot_date(t, theta[:, 3], color = cellc[3], fmt='-')
plt.ylabel(celll[i])
plt.xlabel("Time")
plt.grid(True)
plt.xlim([t[0], t[-1]])

add_logo(f, size_frac_x=1/8, location=1)
plt.tight_layout()
