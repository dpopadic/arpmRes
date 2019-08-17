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

# # S_RollingPrice [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_RollingPrice&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerBondRolPrice).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, array, interp, r_

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlim, scatter, title
import matplotlib.dates as mdates

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict, date_mtop, datenum
from intersect_matlab import intersect
# -

# ## Upload rolling values from 03-Oct-2002 to 03-Oct-2007 with 1 year to maturity, contained in db_SwapCurve

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapCurve'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapCurve'), squeeze_me=True)

DF_Rolling = struct_to_dict(db['DF_Rolling'])

# extraction of rolling values from 03-Oct-2002 to 03-Oct-2007 with tau = 1 year
_, _, dateIndices = intersect([datenum('03-Oct-2002'), datenum('03-Oct-2007')], DF_Rolling.Dates)
_, _, tauIndex = intersect(1, DF_Rolling.TimeToMat)
zroll = DF_Rolling.Prices[tauIndex, dateIndices[0]:dateIndices[1]+1]
dates = DF_Rolling.Dates[dateIndices[0]:dateIndices[1]+1]
time = arange(dates[0],dates[-1]+1)

t_end = array(['03-Oct-2003', '03-Oct-2004', '03-Oct-2005', '03-Oct-2006', '03-Oct-2007'])

_, timeindex, _ = intersect(time, list(map(datenum,t_end)))
# -

# ## Interpolate the rolling value on an yearly spaced grid

zroll = interp(time, dates, zroll[0])

# ## Plot the evolution of the rolling values highlighting them at times t = 03-Oct-2002,...,03-Oct-2006

# rolling value plot
figure()
time_dt = array([date_mtop(i) for i in time])
plot(time_dt, zroll,zorder=1)
scatter(time_dt[timeindex[:-1]], zroll[timeindex[:-1]], marker='.',s=100, c='r',zorder=2)
plt.xticks(time_dt[timeindex])
myFmt = mdates.DateFormatter('%d-%b-%Y')
plt.gca().xaxis.set_major_formatter(myFmt)
xlim([time_dt[0], time_dt[timeindex[-1]]])
plt.grid(True)
title('Rolling prices');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
