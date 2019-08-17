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

# # s_implied_volatility_surface [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_implied_volatility_surface&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerImplVolSurf).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.pricing import implvol_delta2m_moneyness
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_implied_volatility_surface-parameters)

y = 0.02  # yield curve level
l_ = 3  # num. of moneyness points

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_implied_volatility_surface-implementation-step00): Import data

# +
path = '../../../databases/global-databases/derivatives/db_implvol_optionSPX/'
db_impliedvol = pd.read_csv(path + 'data.csv', parse_dates=['date'],
                            keep_date_col=True)
implvol_param = pd.read_csv(path + 'params.csv', index_col=0)

dates = pd.to_datetime(np.array(db_impliedvol.loc[:, 'date']))
t_ = len(dates)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_implied_volatility_surface-implementation-step01): Compute implied volatility surface in the m-moneyness parametrization

# +
tau_implvol = np.array(implvol_param.index)
tau_implvol = tau_implvol[~np.isnan(tau_implvol)]
delta_moneyness = np.array(implvol_param.delta)
k_ = len(tau_implvol)
n_ = len(delta_moneyness)

implied_vol = db_impliedvol.loc[(db_impliedvol['date'].isin(dates)),
                                :].iloc[:, 2:].values

implvol_delta_moneyness_3d = np.zeros((t_, k_, n_))
for k in range(k_):
    implvol_delta_moneyness_3d[:, k, :] = \
        np.r_[np.array(implied_vol[:, k::k_])]

# constant and flat yield curve
y_tau = y*np.ones((t_, k_))

# convert from delta-moneyness to m-moneyness
implvol_m_moneyness, m_moneyness = \
    implvol_delta2m_moneyness(implvol_delta_moneyness_3d, tau_implvol,
                              delta_moneyness, y_tau, tau_implvol, l_)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_implied_volatility_surface-implementation-step02): Save database

# +
implvol2d = implvol_m_moneyness.reshape((t_, k_*l_))
out = pd.DataFrame({'m='+str(m_moneyness[0])+' tau='+str(tau_implvol[0]):
                    implvol2d[:, 0]}, index=dates)
for l in range(1, l_):
    df2 = pd.DataFrame({'m='+str(m_moneyness[l])+' tau='+str(tau_implvol[0]):
                        implvol2d[:, l]}, index=dates)
    out = pd.concat([out, df2], axis=1)
    del df2

for k in range(1, k_):
    for l in range(l_):
        df2 = pd.DataFrame({'m='+str(m_moneyness[l]) +
                            ' tau='+str(tau_implvol[k]):
                            implvol2d[:, k*l_+l]}, index=dates)
        out = pd.concat([out, df2], axis=1)
        del df2

out.index.name = 'dates'
out.to_csv('../../../databases/temporary-databases/db_calloption_rd.csv')
del out
# -

# ## Plots

# +
plt.style.use('arpm')
fig = plt.figure()
X, Y = np.meshgrid(m_moneyness, tau_implvol)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, implvol_m_moneyness[-1, :, :],
                       linewidth=0, antialiased=False)
add_logo(fig)
plt.tight_layout()

fig1 = plt.figure()
label1 = 'Time to expiry '+str(round(tau_implvol[2], 3)) + \
    'y, m_moneyness ' + str(round(m_moneyness[1], 3))
label2 = 'Time to expiry '+str(round(tau_implvol[4], 3)) + \
    'y, m_moneyness ' + str(round(m_moneyness[1], 3))
plt.plot(implvol_m_moneyness[:, 2, 1], 'r')
plt.plot(implvol_m_moneyness[:, 4, 1])
plt.gca().legend((label1, label2))
add_logo(fig1)
plt.tight_layout()

fig2 = plt.figure()
X, Y = np.meshgrid(delta_moneyness, tau_implvol)
ax = fig2.gca(projection='3d')
surf = ax.plot_surface(X, Y, implvol_delta_moneyness_3d[-1, :, :],
                       linewidth=0, antialiased=False)
add_logo(fig2)
plt.tight_layout()

fig3 = plt.figure()
label1 = 'Time to expiry '+str(round(tau_implvol[2], 3)) + \
    'y, delta_moneyness ' + str(round(delta_moneyness[1], 3))
label2 = 'Time to expiry '+str(round(tau_implvol[4], 3)) + \
    'y, delta_moneyness ' + str(round(delta_moneyness[1], 3))
plt.plot(implvol_delta_moneyness_3d[:, 2, 1], 'r')
plt.plot(implvol_delta_moneyness_3d[:, 4, 1])
plt.gca().legend((label1, label2))
add_logo(fig3)
plt.tight_layout()
