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

# # s_volatility_clustering_stock [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_volatility_clustering_stock&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerVolClust).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.statistics import invariance_test_ellipsoid
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_volatility_clustering_stock-parameters)

l_ = 25  # number of lags (for autocorrelation test)
conf_lev = 0.95

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_volatility_clustering_stock-implementation-step00): Upload data

path = '../../../databases/global-databases/equities/db_stocks_SP500/'
db_stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0], index_col=0)
db_stocks.index = pd.to_datetime(db_stocks.index)
dates =  pd.to_datetime(db_stocks.index)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_volatility_clustering_stock-implementation-step01): Compute risk drivers for GE, compounded return and its absolute value

# +
# risk drivers
log_underlying = np.log(np.array(db_stocks.loc[dates, 'GE']))

# compounded return and its absolute value
comp_return = np.diff(log_underlying)
abs_comp_return = np.abs(comp_return)
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(15.5, 9))
plt.plot(dates[1:], comp_return.flatten(), '*', markersize=3)
plt.title('Compounded return')
add_logo(fig, set_fig_size=False)

plt.figure()
acf_x, conf_int_x = \
    invariance_test_ellipsoid(comp_return, l_, conf_lev=conf_lev,
                              title='Compounded return', plot_test=True)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)

plt.figure()
acf_delta_x, conf_int_detal_x = \
    invariance_test_ellipsoid(abs_comp_return, l_, conf_lev=conf_lev, bl=-0.025,
                              title='Absolute compounded return', plot_test=True)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)
