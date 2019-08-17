#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script performs the Kolmogorov-Smirnov test for invariance on four
# different variables, computed from the dividend adjusted prices of one
# stock.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=iidtest-equity-copy-1).

# +
# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB

from ARPM_utils import save_plot, struct_to_dict
from TestKolSmirn import TestKolSmirn
from InvarianceTestKolSmirn import InvarianceTestKolSmirn
# -

# ## Upload the database
# ## Upload daily stock prices from db_Stocks

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stocks'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stocks'), squeeze_me=True)

StocksSPX = struct_to_dict(db['StocksSPX'])
# -

# ## Compute the dividend adjusted prices of one stock

stock_index = 0
v = StocksSPX.Prices[[stock_index], :]
date = StocksSPX.Date
div = StocksSPX.Dividends[stock_index][0]
# if not div:
#     v,_=Price2AdjustedPrice(date,v,div)

# ## Compute the time series for each variable

x = v[[0], 1:] / v[[0], :-1]
y = v[[0], 1:] - v[[0], :-1]
z = (v[[0], 1:] / v[[0], :-1]) ** 2
w = v[[0], 2:] - 2*v[[0], 1:-1] + v[[0], :-2]

# ## Compute the Kolmogorov-Smirnov test for each variable

x_1, x_2, band_x, F_1_x, F_2_x, up_x, low_x = TestKolSmirn(x)
y_1, y_2, band_y, F_1_y, F_2_y, up_y, low_y = TestKolSmirn(y)
z_1, z_2, band_z, F_1_z, F_2_z, up_z, low_z = TestKolSmirn(z)
w_1, w_2, band_w, F_1_w, F_2_w, up_w, low_w = TestKolSmirn(w)

# ## Create figures showing the results of Kolmogorov-Smirnov test

# +
# x
f = figure()
InvarianceTestKolSmirn(x, x_1, x_2, band_x, F_1_x, F_2_x, up_x, low_x, [], 'Invariance Test (X)');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# y
f = figure()
InvarianceTestKolSmirn(y, y_1, y_2, band_y, F_1_y, F_2_y, up_y, low_y, [], 'Invariance Test (Y)');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# z
f = figure()
InvarianceTestKolSmirn(z, z_1, z_2, band_z, F_1_z, F_2_z, up_z, low_z, [], 'Invariance Test (Z)');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# w
f = figure()
InvarianceTestKolSmirn(w, w_1, w_2, band_w, F_1_w, F_2_w, up_w, low_w, [], 'Invariance Test (W)');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

