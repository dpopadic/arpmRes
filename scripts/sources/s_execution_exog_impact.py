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

# # s_execution_exog_impact [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_execution_exog_impact&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-exog-impact).

# +
import numpy as np
import pandas as pd

from scipy.special import erf

from arpym.estimation import fit_lfm_ols
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_execution_exog_impact-parameters)

gamma = 5  # constant for rescaled error function
l = 10  # number of lags included in the model

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_execution_exog_impact-implementation-step00): Load data

path = '../../../databases/global-databases/high-frequency/db_stocks_highfreq/'
msft = pd.read_csv(path + 'MSFT/data.csv', index_col=0, parse_dates=True)
p = np.array(msft.loc[:, 'trade_price'])
p_ask = np.array(msft.loc[:, 'ask'])
p_bid = np.array(msft.loc[:, 'bid'])
delta_q = np.array(msft.loc[:, 'trade_size'])

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_execution_exog_impact-implementation-step01): Compute the realizations of the variable sign

# +
# take data with (ask > bid) and (price = bid or price = ask)
index = np.where((p_ask > p_bid) & ((p == p_bid) | (p == p_ask)))

frac = (p[index] - p_bid[index]) / (p_ask[index] - p_bid[index])
sgn = erf(gamma*(2*frac - 1))  # sign time series
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_execution_exog_impact-implementation-step02): Compute the realization of the fair price, signed volume and price changes

mid_quote = (p_bid[index] + p_ask[index]) / 2  # mid-quote time series
delta_sgn_q = sgn * delta_q[index].astype(float)  # signed-volume time series
delta_p = np.diff(mid_quote)  # price changes time series

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_execution_exog_impact-implementation-step03): construction of the lagged variables (lagged traded volumes with sign)

# +
delta_p_lag = delta_p[l:]  # lagged variable delta_p
d_ = len(delta_p_lag)
delta_sgn_q = delta_sgn_q[1:]  # drop first in order to match dimensions

delta_sgn_q_lag = np.zeros((d_, l))

for i in range(1, l+1):  # lagged variable delta_sgn_q
    delta_sgn_q_lag[:, i-1] = delta_sgn_q[l - i: -i]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_execution_exog_impact-implementation-step04): Fit the model via OLS regression

p_fp = np.ones((d_,)) / d_  # flat flexible probabilities
_, b, _, _ = fit_lfm_ols(delta_p_lag, delta_sgn_q_lag, p_fp)
print(b)  # print the fitted loadings
