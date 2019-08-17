#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_kolmsmirn_ytm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_kolmsmirn_ytm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exerd-yinv-copy-1).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics import invariance_test_ks
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_ytm-parameters)

t_ = 1000  # length of time series of yields
tau = 10  # selected time to maturity (years)
conf_lev = 0.95  # confidence level

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_ytm-implementation-step00): Load data

tau = np.array([tau])
path = '../../../databases/global-databases/fixed-income/db_yields'
y = pd.read_csv(path + '/data.csv', header=0, index_col=0)
y = y[tau.astype(float).astype(str)].tail(t_).values.reshape(-1)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_ytm-implementation-step01): Compute the daily yield increment

epsi = np.diff(y)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_ytm-implementation-step02): Perform Kolmogorov-Smirnov test

# +
plt.style.use('arpm')

# perform and show Kolmogorov-Smirnov test for invariance
z_ks, z = invariance_test_ks(epsi, conf_lev=conf_lev)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)
