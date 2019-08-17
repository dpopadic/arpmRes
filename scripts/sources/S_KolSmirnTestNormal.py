#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script performs the Kolmogorov-Smirnov test for invariance on
# simulations of a normal random variable.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-iidtests-copy-1).

# +
# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from ARPM_utils import save_plot
from TestKolSmirn import TestKolSmirn
from InvarianceTestKolSmirn import InvarianceTestKolSmirn

# input parameters
t_ = 1000  # time series len
mu = 0  # expectation
sigma = 0.25  # standard deviation
# -

# ## Generate normal simulations

Epsi = norm.rvs(mu, sigma, (1, t_))

# ## Perform Kolmogorov-Smirnov test

Epsi_1, Epsi_2, band_int, F_1, F_2, up_band, low_band = TestKolSmirn(Epsi)

# ## Plot Kolmogorov-Smirnov test for invariance

f = figure()
InvarianceTestKolSmirn(Epsi, Epsi_1, Epsi_2, band_int, F_1, F_2, up_band, low_band);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
