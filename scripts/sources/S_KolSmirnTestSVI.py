#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 'his script performs the Kolmogorov-Smirnov test for invariance on the
# parameter increments of SVI model.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-sviiid-copy-1).

# +
# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import diff

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from TestKolSmirn import TestKolSmirn
from InvarianceTestKolSmirn import InvarianceTestKolSmirn
# -

# ## Load the database generated by script S_FitSVI

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_FitSVI'))
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_FitSVI'))

theta = db['theta']
# -

# ## Compute increments and perform Kolmogorov-Smirnov test

# +
# initialize variables
delta_theta = {}
s1 = {}
s2 = {}
int = {}
F_s1 = {}
F_s2 = {}
up = {}
low = {}

for k in range(6):
    delta_theta[k] = diff(theta[k, :]).reshape(1, -1)  # increments
    [s1[k], s2[k], int[k], F_s1[k], F_s2[k], up[k], low[k]] = TestKolSmirn(delta_theta[k])  # Kolmogorov-Smirnov test
# -

# ## Plot the results of the IID test

# +
# position settings
pos = {}
pos[0] = [0.1300, 0.74, 0.3347, 0.1717]
pos[1] = [0.5703, 0.74, 0.3347, 0.1717]
pos[2] = [0.1300, 0.11, 0.7750, 0.5]
pos[3] = [0.15, 1.71]
# names of figures
name = {}
name[0] = r'Invariance test (increments  of $\theta_1$)'
name[1] = r'Invariance test (increments  of $\theta_2$)'
name[2] = r'Invariance test (increments  of $\theta_3$)'
name[3] = r'Invariance test (increments  of $\theta_4$)'
name[4] = r'Invariance test (increments  of $\theta_5$)'
name[5] = r'Invariance test (increments  of $\theta_6$)'

for k in range(6):
    f = figure()
    InvarianceTestKolSmirn(delta_theta[k], s1[k], s2[k], int[k], F_s1[k], F_s2[k], up[k], low[k], pos, name[k]);
    # save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])