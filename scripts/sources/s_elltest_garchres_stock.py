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

# # s_elltest_garchres_stock [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_elltest_garchres_stock&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerGarchFigure).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation import fit_garch_fp
from arpym.statistics import invariance_test_ellipsoid
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_garchres_stock-parameters)

l_ = 25  # number of lags (for autocorrelation test)
conf_lev = 0.95

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_garchres_stock-implementation-step00): Upload data

path = '../../../databases/global-databases/equities/db_stocks_SP500/'
db_stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0], index_col=0)
db_stocks.index = pd.to_datetime(db_stocks.index)
dates =  pd.to_datetime(db_stocks.index)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_garchres_stock-implementation-step01): Compute risk drivers for GE and compounded return

# +
# risk drivers
log_underlying = np.log(np.array(db_stocks.loc[dates, 'GE']))

# compounded return
comp_return = np.diff(log_underlying)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_garchres_stock-implementation-step02): Fit GARCH(1,1), compute residuals and their absolute values

# +
_, _, epsi = fit_garch_fp(comp_return)

abs_epsi = np.abs(epsi)
# -

# ## Plots

fig = plt.figure()
acf_x, conf_int_x = \
    invariance_test_ellipsoid(abs_epsi, l_, conf_lev=conf_lev, bl=-0.75,
                              title='Absolute residuals of a GARCH(1, 1) model fitted on stock compounded return',
                              plot_test=True)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)
