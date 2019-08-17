#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# variables related to weekly observed implied volatility, namely:
#  - weekly changes in implied volatility
#  - weekly changes in log implied volatility
#  - residuals of a multivariate autoregressive fit of order one by means
#   of least squares method.
# The results are then plotted in three different figures.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=iidtest-implied-vol-copy-1).

# +
# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import reshape, ones, squeeze, diff, \
    eye, log, r_
from numpy.linalg import solve

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB
from ARPM_utils import save_plot
from TestKolSmirn import TestKolSmirn
from InvarianceTestKolSmirn import InvarianceTestKolSmirn
# -

# ## Upload database

db = loadmat(os.path.join(GLOBAL_DB, 'db_Derivatives'))
Sigma = db['Sigma']

# ## Select weekly observations of implied volatility

delta_t = 5
sigma = Sigma[:, :, ::delta_t]

# ## Perform the Kolmogorov-Smirnov test on weekly changes in implied vol

# +
tau_index = 1  # time to maturity index
m_index = 4  # moneyness index

delta_sigma = diff(squeeze(sigma[tau_index, m_index, :])).reshape(1, -1)  # changes in implied volatility
s1, s2, int, F1, F2, up, low = TestKolSmirn(delta_sigma)
# -

# ## Perform the Kolmogorov-Smirnov test on weekly changes in log implied vol

log_sigma = log(squeeze(sigma[tau_index, m_index, :]))  # logarithm of implied vol
delta_log_sigma = diff(log_sigma).reshape(1, -1)  # changes in log implied volatility
s1_log, s2_log, int_log, F1_log, F2_log, up_log, low_log = TestKolSmirn(delta_log_sigma)

# ## Perform the least squares fitting and the Kolmogorov-Smirnov test on residuals

# +
tau_, m_, t_ = sigma.shape
sigma = reshape(sigma, (tau_ * m_, t_))

y = sigma[:, 1:].T
x = r_['-1', ones((t_ - 1, 1)), sigma[:, :-1].T]

yx = y.T@x
xx = x.T@x
b = yx@solve(xx, eye(xx.shape[0]))
r = y - x@b.T  # residuals

epsi = r[:, [2]].T  # select the residuals corresponding to 60 days-to-maturiy and moneyness equal to 0.9
s1_res, s2_res, int_res, F1_res, F2_res, up_res, low_res = TestKolSmirn(epsi)
# -

# ## Plot the results of the IID test

# +
pos = {}
pos[1] = [0.1300, 0.74, 0.3347, 0.1717]
pos[2] = [0.5703, 0.74, 0.3347, 0.1717]
pos[3] = [0.1300, 0.11, 0.7750, 0.5]
pos[4] = [0.03, 1.71]

f = figure()  # changes in implied vol
InvarianceTestKolSmirn(delta_sigma, s1, s2, int, F1, F2, up, low, pos,
                       'Kolm.-Smir. test on weekly increments of implied volatility');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

f = figure()  # changes in log implied vol
InvarianceTestKolSmirn(delta_log_sigma, s1_log, s2_log, int_log, F1_log, F2_log, up_log, low_log, pos,
                       'Kolm.-Smir. test on weekly increments of log implied volatility');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

f = figure()  # residuals of the autoregressive fitting
InvarianceTestKolSmirn(epsi, s1_res, s2_res, int_res, F1_res, F2_res, up_res, low_res, pos,
                       'Kolm.-Smir. test on residuals of autoregressive fit');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
