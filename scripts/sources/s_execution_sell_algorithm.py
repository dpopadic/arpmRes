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

# # s_execution_sell_algorithm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_execution_sell_algorithm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-execution_-sell-algorithm).

import numpy as np

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_execution_sell_algorithm-parameters)

delta_h_child = -100  # shares to sell
delta_h_residual = np.array([delta_h_child])
t_end = 60000  # end of the execution time interval in milliseconds
kappa_ = 40  # effective number of ticks in the interval
e_kleft = np.array([30])  # initial the expectation on the number of ticks
delta_h = np.array([])

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_execution_sell_algorithm-implementation-step01): Wall-clock time series corresponding to the randomly generated kappa_ ticks

t = np.sort(np.random.randint(t_end, size=(kappa_,)))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_execution_sell_algorithm-implementation-step02): Proceed with the algorithm until the order is fully executed

for kappa in range(kappa_):
    # compute the sell order
    delta_h = np.append(delta_h,
                        round(delta_h_residual[kappa] / e_kleft[kappa]))
    # review the expectation on the residual tick time
    e_kleft = np.append(e_kleft,
                        round((kappa+1)*(t_end - t[kappa]) / t[kappa]))
    # compute the residual amount to be sold
    delta_h_residual = np.append(delta_h_residual,
                                 delta_h_residual[kappa] - delta_h[kappa])
    # break if order is fully executed
    if delta_h_residual[kappa+1] == 0:
            break
    # place an order to sell residual amount
    if e_kleft[kappa+1] == 0 or kappa == kappa_-1:
        delta_h = np.append(delta_h, delta_h_residual[kappa+1])
        break

# ## Displays

# +

for kappa in range(len(delta_h_residual)-1):
    print('k = {kappa} : place a market order to sell {dtick} units'
          ' at the best bid at the {tot}th millisecond, remains {remunits}'
          .format(kappa=kappa+1, dtick=abs(delta_h[kappa]),
                  tot=np.squeeze(t[kappa]),
                  remunits=abs(delta_h_residual[kappa+1])))
if delta_h_residual[-1] < 0:
    print('Place a market order to sell the remaining {dtick} units'
          ' at the best bid at the end of the minute'
          .format(dtick=abs(delta_h[-1])))
