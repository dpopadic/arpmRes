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

# # s_currency_carry [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_currency_carry&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-carry-in-carrencies).

# +
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_currency_carry-parameters)

k = 1 / 116.5  # strike (forward rate at inception)
fx_jpy_us = 1 / 114.68  # spot yen to dollar on 26-Dec-2013
time_to_mat = 1  # maturity of the contract
dt = 1 / 100

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_currency_carry-implementation-step01): key rates for USD and JGB yield curves at the current time

# +
tam_l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40]
steady_path_l = [0.00, 0.080, 0.098, 0.122, 0.173, 0.221, 0.316, 0.443, 0.552,
                 0.635, 0.707, 1.104, 1.557, 1.652, 1.696,
                 1.759]  # JGB yield curve on 26-Dec-2013

tam_b = [0, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30]
steady_path_b = [0.00, 0.07, 0.09, 0.13, 0.42, 0.81, 1.74,
                 2.43, 3.00, 3.68, 3.92]  # USD yield curve on 26-Dec-2013
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_currency_carry-implementation-step02): carry of the forward contract

# +
# initial settings
t_hor = np.arange(0, 1 + dt, dt)
m_ = len(t_hor)

maturities = np.arange(time_to_mat, -dt, -dt)  # maturities

interp = interpolate.interpolate.interp1d(tam_l, steady_path_l,
                                          fill_value='extrapolate')
yield_curve_l = interp(maturities)
interp = interpolate.interpolate.interp1d(tam_b, steady_path_b,
                                          fill_value='extrapolate')
yield_curve_b = interp(maturities)

carry_thor = -(fx_jpy_us * np.exp((-maturities * yield_curve_l)) -
               k * np.exp((-maturities * yield_curve_b)) -
               fx_jpy_us * np.exp(-time_to_mat * yield_curve_l) +
               k * np.exp(-time_to_mat * yield_curve_b))
# -

# ## Plots

# +
plt.style.use('arpm')

fig, ax = plt.subplots(2, 1)
nu = 0.5  # nu=u-t
i = np.where(t_hor == nu)[0][0]
dgrey = [0.2, 0.2, 0.2]  # dark grey
lgrey = [0.6, 0.6, 0.6]  # light grey
blue = [0, 0, 0.4]  # dark blue
plt.sca(ax[0])
plt.grid(True)
plt.axis([0, 1, np.min(carry_thor), max(carry_thor)])
plt.xticks(np.arange(maturities[i], 1, 0.1), np.arange(0, nu + 0.1, 0.1))
shift_carry = carry_thor[:i+1].reshape(1, -1)
shift_time = t_hor[i:].reshape(1, -1)
plt.plot([maturities[i], maturities[i]], [np.min(carry_thor) - 0.05,
         np.max(carry_thor) + 0.2], color='k', lw=1)
plt.plot(np.r_[shift_time, shift_time], np.r_[np.zeros(shift_carry.shape),
         shift_carry], color=lgrey, lw=1)
plt.plot(shift_time[0], shift_carry[0], color=lgrey, lw=1)
for i in range(shift_time.shape[1]-1):
    plt.fill_between([shift_time[0, i], shift_time[0, i+1]],
                     [shift_carry[0, i], 0], facecolor=lgrey, edgecolor=lgrey)
plt.xlabel('Time (years)')
plt.ylabel('Carry')
plt.title('Carry in currencies')
# Yield to maturity curves
plt.sca(ax[1])
plt.axis([0, 1, min(np.min(yield_curve_b), np.min(yield_curve_l)),
          max(np.max(yield_curve_b), np.max(yield_curve_l))])
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid(True)
# yield curve (base currency)
plt.plot(maturities, yield_curve_b, color=blue, lw=1)
plt.plot([maturities[i], maturities[i]], [yield_curve_b[i], yield_curve_b[i]],
         color='r', marker='.', markersize=15)
plt.text(maturities[i], yield_curve_b[i - 15] + 0.002, '$y_{t}^{b}$')
# yield curve (local currency)
plt.plot(maturities, yield_curve_l, color=blue, lw=1)
plt.plot([maturities[i], maturities[i]], [yield_curve_l[i], yield_curve_l[i]],
         color='r', marker='.', markersize=15)
plt.text(maturities[i], yield_curve_l[i - 15] + 0.002, '$y_{t}^{l}$')
plt.xlabel('Time to Maturity (years)')
plt.ylabel('Yield')
plt.title('Yield to maturity curves')
add_logo(fig)
plt.tight_layout()
