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

# # s_yield_curve_evolution [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_yield_curve_evolution&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerSwapCurve).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_yield_curve_evolution-parameters)

t_ = 500  # length of time series of yields
tau_select = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # selected times to maturity
tau_select = np.sort(tau_select)

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_yield_curve_evolution-implementation-step00): Load data

path = '../../../databases/global-databases/fixed-income/db_yields/'
df_data = pd.read_csv(path + 'data.csv', header=0, index_col=0,
                             parse_dates=True, infer_datetime_format=True)
df_data = df_data.iloc[-t_:]

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_yield_curve_evolution-implementation-step01): Select data

df_data = df_data.loc[:, str(float(tau_select[0])):str(float(tau_select[-1]))]
# extract values
t = np.array(df_data.index, dtype=np.datetime64)
tau = df_data.columns.values.astype(np.float32)
y_t_tau = df_data.values

# ## Plots

# +
min_date = np.min(t)
t_relative = np.int32((t - min_date).astype('timedelta64[D]'))
plt.style.use('arpm')
t_mesh, tau_mesh = np.meshgrid(t_relative, tau)
no_of_xticks = 6
xticks = t[np.linspace(0, t.size-1, no_of_xticks, dtype=int)]
xticks = pd.to_datetime(xticks).strftime('%d-%b-%Y')
xticks_location = t_relative[np.linspace(0, t.size-1, no_of_xticks, dtype=int)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(t_mesh, tau_mesh, y_t_tau.T, rcount=10, ccount=37)
ax.plot(np.ones(tau.size)*t_relative[-1], tau, y_t_tau.T[:, -1], lw=1.5, c='r')
for x in tau_select:
    index, = np.where(x == tau)[0]
    ax.plot(t_relative, np.ones_like(t_relative)*x, y_t_tau.T[index, :],
            lw=1,
            color=(0.9, 0.9, 0.9))
ax.view_init(22, -67)
ax.set_xticks(xticks_location)
ax.set_xticklabels(xticks, rotation=10)
ax.set_xlabel('Time', labelpad=-15)
ax.set_ylabel('Time to Maturity', labelpad=8)
ax.set_yticklabels([str(x)+' y' for x in ax.get_yticks()])
ax.set_zlabel('Yield', rotation=90, labelpad=8)
ax.set_zticklabels([str(x)+' %' for x in ax.get_zticks()*100])
plt.title('Swap curve', fontweight='bold')
add_logo(fig)
plt.tight_layout()
