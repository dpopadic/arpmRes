#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_bond_carry [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bond_carry&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-carry-cb).

# +
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from arpym.pricing import bond_value, cash_flow_reinv
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bond_carry-parameters)

t_now = np.datetime64('2011-06-27')  # current date
tau_hor = 108  # time to horizon
c = 0.04  # annualized coupons (percentage of the face value)
freq_paym = 1  # coupon payment frequency (years)

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_bond_carry-implementation-step00): Upload data

path = '../../../databases/temporary-databases'
tau = np.array([1, 2, 3, 5, 7, 10, 15, 30])  # times to maturity
path = '../../../databases/global-databases/fixed-income/db_yields'
y = pd.read_csv(path + '/data.csv', header=0, index_col=0)
# select the yields corresponding to current time
y = y[tau.astype(float).astype(str)]
y_carry = y.loc[y.index == pd.to_datetime(str(t_now)).strftime("%d-%b-%Y")]

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bond_carry-implementation-step01): Monitoring dates, record dates and coupons of the bond

# +
t_end = np.datetime64('2025-12-22')  # maturity date
m_ = tau_hor

# monitoring dates
deltat_m = 21
t_m = np.busday_offset(t_now, np.arange(m_+1)*deltat_m, roll='forward')

# # number of coupons until bond maturity
k_ = int(np.busday_count(t_m[0], t_end)/(freq_paym*252))

# record dates
r = np.busday_offset(t_now, np.arange(1, k_+1)*int(freq_paym*252))

# coupons
coupon = c * freq_paym * np.ones(len(r))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_bond_carry-implementation-step02): dirty price appreciation carry

# +
v_t_hor = np.array([bond_value(eval_t, coupon, r, 'y', y_carry, tau_x=tau)
                   for eval_t in t_m]).T

carry_dirty_price = (v_t_hor - v_t_hor.reshape(-1)[0]).reshape(-1)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_bond_carry-implementation-step03): Reinvested cash flows contribution component of carry

# +
# reinvestment factor
interp = sp.interpolate.interp1d(tau.flatten(), y_carry,
                                 axis=1, fill_value='extrapolate')
y_0 = interp(0)

invfact_m = np.ones((1, m_)) * np.exp(deltat_m*y_0/252)

# include notional with last coupon
coupon[-1] = coupon[-1] + 1

# cash flow stream

cf_t_hor = cash_flow_reinv(coupon, r, t_m, invfact_m)
cf_t_hor = cf_t_hor.reshape(-1)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_bond_carry-implementation-step04): Bond carry from current time to horizon

carry_t_now_t_hor = carry_dirty_price + np.r_[0.0, cf_t_hor]

# ## Plots

# +
plt.style.use('arpm')
fig, ax = plt.subplots(2, 1)


dgrey = [0.2, 0.2, 0.2]  # dark grey
lgrey = [0.6, 0.6, 0.6]  # light grey
blue = [0, 0, 0.4]  # dark blue

plt.sca(ax[0])
plt.grid(True)

time = [np.busday_count(t_now, t_m[i])/252 for i in range(m_+1)]
plt.plot([0, 0], [min(carry_t_now_t_hor), 0.6], color='k')
l1 = plt.plot([time, time], [carry_t_now_t_hor, np.zeros(m_+1)], color=dgrey,
              lw=2)
l2 = plt.plot([time[1:], time[1:]], [cf_t_hor, np.zeros(m_)], color=lgrey, lw=2)
plt.axis([-np.busday_count(t_now, t_end)/252+tau_hor/12 - 0.1,
          tau_hor/12 + 0.1, 0, 0.6])
plt.xticks(np.arange(0, tau_hor/12 + 1, 1))
plt.legend(handles=[l1[0], l2[0]], labels=['price', 'coupon'])
plt.xlabel('Time (years)')
plt.ylabel('Carry')
plt.title('Coupon bond carry')

# bottom plot
plt.sca(ax[1])
time1 = np.arange(0, np.busday_count(t_now, t_end)/252+0.1, 0.1)
yield_curve = interp(time1)

plt.plot(time1, yield_curve.reshape(-1), color=blue)  # yield curve
yield_t_hor = interp((np.busday_count(t_now, t_end)/21-tau_hor)/12)
plt.plot((np.busday_count(t_now, t_end)/21-tau_hor)/12, yield_t_hor, color='r',
         marker='.', markersize=15)
plt.xlim(-0.1, np.busday_count(t_now, t_end)/252 + 0.1)
plt.xticks(np.arange(0, np.busday_count(t_now, t_end)/252 + 1, 1))

plt.xlabel('Time to Maturity (years)')
plt.ylabel('Yield')
plt.title('Yield to maturity curve')

add_logo(fig)
plt.tight_layout()
