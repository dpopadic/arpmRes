#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_pricing_calloption [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pricing_calloption&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-call-option-value).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from tqdm import trange

from arpym.statistics import meancov_sp
from arpym.pricing import call_option_value, ytm_shadowrates
from arpym.tools import add_logo, histogram_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_calloption-parameters)

tau_hor = 100  # time to horizon (in days)
j_ = 1000  # number of scenarios
k_strk = 1407  # strike of the options on the S&P500 (in dollars)
t_end = np.datetime64('2013-08-31')  # expiry date of the options
y = 0.02  # yield curve (assumed flat and constant)

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_calloption-implementation-step00): Import data

# +
path = '../../../databases/temporary-databases/'

db_proj = pd.read_csv(path+'db_calloption_proj.csv', index_col=0)
m_moneyness = np.array([float(col[col.find('m=')+2:col.find(' tau=')])
                        for col in db_proj.columns[1:]])
m_moneyness = np.unique(m_moneyness)
tau_implvol = np.array([float(col[col.find(' tau=')+5:])
                        for col in db_proj.columns[1:]])
tau_implvol = np.unique(tau_implvol)
db_projdates = pd.read_csv(path + 'db_calloption_proj_dates.csv', header=0,
                           parse_dates=True)
t_m = np.array(pd.to_datetime(db_projdates.values.reshape(-1)),
               dtype='datetime64[D]')
m_ = t_m.shape[0]-1
deltat_m = np.busday_count(t_m[0], t_m[1])
if tau_hor > m_:
    print(" Projection doesn't have data until given horizon!!!" +
          " Horizon lowered to ", m_)
    tau_hor = m_
# number of monitoring times
m_ = tau_hor
t_m = t_m[:m_+1]
i_ = db_proj.shape[1]
x_proj = db_proj.values.reshape(j_, -1, i_)
x_proj = x_proj[:, :m_+1, :]
x_tnow = x_proj[0, 0, :]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_calloption-implementation-step01): Pricing at the horizon

# +
v_call_thor = np.zeros((j_, m_+1))
log_sigma_atm = np.zeros((j_, m_+1))
s_thor = np.zeros((j_, m_+1))

points = list(zip(*[grid.flatten() for grid in
                    np.meshgrid(*[tau_implvol, m_moneyness])]))
for m in trange(m_+1,desc='Day'):
    tau = np.busday_count(t_m[m], t_end)/252
    if tau < tau_implvol[0]:
        tau = tau_implvol[0]
    for j in range(j_):
        # compute shadow yield
        x_y = ytm_shadowrates(np.array([y]))
        x_y = np.atleast_1d(x_y)
        # compute call option value
        v_call_thor[j, m] = \
            call_option_value(x_proj[j, m, 0], x_y, tau,
                              x_proj[j, m, 1:], m_moneyness, tau_implvol,
                              k_strk, t_end, t_m[m])
        # compute log-implied volatility at the moneyness
        log_sigma_atm[j, m] = \
            interpolate.LinearNDInterpolator(points,
                                             x_proj[j, m, 1:])(*np.r_[tau, 0])
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_calloption-implementation-step02): Scenario-probability expectations and standard deviations

# +
mu_v = np.zeros(m_+1)
sig_v = np.zeros(m_+1)

for m in range(len(t_m)):
    mu_v[m], sig1 = meancov_sp(v_call_thor[:, m].reshape(-1, 1))
    sig_v[m] = np.sqrt(sig1)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_calloption-implementation-step03): Save databases

# +
output = {'j_': pd.Series(j_),
          'k_strike': pd.Series(k_strk),
          't_end': pd.Series(t_end),
          'm_': pd.Series(m_),
          'y_rf': pd.Series(y),
          't_m': pd.Series(t_m),
          'log_s': pd.Series(x_proj[:, :, 0].reshape((j_*(m_+1),))),
          'v_call_thor': pd.Series(v_call_thor.reshape((j_*(m_+1),))),
          'log_sigma_atm': pd.Series(log_sigma_atm.reshape((j_*(m_+1),)))}

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_call_data.csv')
# -

# ## Plots

# +
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey

num_plot = min(j_, 20)
fig = plt.figure()

plt.xlim([0, m_+int(m_/3)])
for j in range(num_plot):
    plt.plot(np.arange(0, m_+1), v_call_thor[j, :], lw=1, color=lgrey)

l2 = plt.plot(np.arange(m_+1), mu_v+sig_v, 'r')
plt.plot(np.arange(m_+1), mu_v-sig_v, 'r')
l1 = plt.plot(np.arange(0, m_+1), mu_v, 'g')

y_hist, x_hist = histogram_sp(v_call_thor[:, m_], k_=50*np.log(j_))
y_hist = y_hist*2500
shift_y_hist = m_ + y_hist
# # empirical pdf
pdf = plt.barh(x_hist, y_hist, (max(x_hist)-min(x_hist))/(len(x_hist)-1),
               left=m_, facecolor=lgrey, edgecolor=lgrey,
               lw=2, label='horizon pdf')
plt.plot(shift_y_hist, x_hist, color=dgrey, lw=1)
plt.legend(handles=[l1[0], l2[0], pdf[0]],
           labels=['mean', ' + / - st.deviation', 'horizon pdf'])
plt.title("Call option projected value at the horizon")
add_logo(fig)
fig.tight_layout()

fig2 = plt.figure()

plt.scatter(x_proj[:, -1, 0], v_call_thor[:, -1], 3, np.array([dgrey]), '*')

add_logo(fig2)
fig2.tight_layout()
# -


