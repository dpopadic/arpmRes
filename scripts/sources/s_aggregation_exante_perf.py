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

# # s_aggregation_exante_perf [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_aggregation_exante_perf&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EXAggreVarAndGenObj).

import numpy as np

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_exante_perf-parameters)

h = np.array([30, 3000, 10000, -10000])  # portfolio holdings
cash_t = 2000  # cash at time t invested in the portfolio
v_t = np.array([150, 0.8, 0.6, 0.14])  # values of the instruments at time t
pl = np.array([-5.55, 0.05, 0.2, 0.1])  # unit P&L's over [t,u)
d = np.array([150, 1, 0.8, 0.5])  # basis values associated to the instruments
h_b = np.array([20, 3000, 3750, 6000])  # benchmark portfolio holdings
cashb_t = 3510  # cash at time t invested in the benchmark portfolio

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_exante_perf-implementation-step01): Compute the portfolio P&L, value and return

pl_h = h.T@pl
v_h = cash_t + h.T@v_t
r_h = pl_h / v_h

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_exante_perf-implementation-step02): Compute the benchmark portfolio P&L, value and return

pl_b = h_b.T@pl
v_b = cashb_t + h_b.T@v_t
r_b = pl_b / v_b

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_exante_perf-implementation-step03): Show that the aggregation rule for the portfolio return holds

w = h * d / v_h  # instruments weights
r = pl / d  # generalized returns of the instruments in the portfolio
r_h_2 = w.T@r  # portfolio return

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_exante_perf-implementation-step04): Compute the excess generalized return

excess_r = r_h - r_b

# ##  Step 5: Compute the excess generalized return as a function of the instruments P&L's

h_tilde_ret = h / v_h - h_b / v_b  # standardized holdings
excess_r_2nd = h_tilde_ret.T@pl

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_aggregation_exante_perf-implementation-step06): Compute the excess generalized return as a function of the instruments returns

w_b = h_b * d / v_b  # instruments weights in the benchmark portfolio
excess_r_3rd = (w - w_b).T@r
