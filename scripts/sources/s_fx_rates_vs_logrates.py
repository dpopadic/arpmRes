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

# # s_fx_rates_vs_logrates [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fx_rates_vs_logrates&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerDriversFX).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.tools import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fx_rates_vs_logrates-implementation-step00): Upload data

fx_path = '../../../databases/global-databases/currencies/db_fx/data_long.csv'
fx_df = pd.read_csv(fx_path, usecols=['date', 'spot_usd_gbp'],
                    parse_dates=['date'])
fx_usd2gbp = fx_df.spot_usd_gbp  # USD/GBP exchange rate

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fx_rates_vs_logrates-implementation-step01): Compute the time series of the inverse exchange rate

fx_gbp2usd = 1 / fx_usd2gbp  # GBP/USD exchange rate

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fx_rates_vs_logrates-implementation-step02): Compute the time series of the log-exchange rate and its opposite

log_fx_usd2gbp = np.log(fx_usd2gbp)  # USD/GBP log-exchange rate
log_fx_gbp2usd = -log_fx_usd2gbp  # GBP/USD log-exchange rate

# ## Plots

# +
plt.style.use('arpm')

fig, axs = plt.subplots(2, 1)

axs[0].plot(fx_df.date, fx_usd2gbp)
axs[0].plot(fx_df.date, fx_gbp2usd)
axs[0].set_title('FX USD-GBP')
axs[0].legend(['FX', '1/FX'])

axs[1].plot(fx_df.date, log_fx_usd2gbp)
axs[1].plot(fx_df.date, log_fx_gbp2usd)
axs[1].legend(['log-FX', '- log-FX'])
add_logo(fig, location=6)
plt.tight_layout()
