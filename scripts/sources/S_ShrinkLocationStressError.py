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

# # S_ShrinkLocationStressError [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ShrinkLocationStressError&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerShrinkStar).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import zeros, arange, mean, argmin, argmax, max as npmax, min as npmin

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, bar, xlim, ylim, title, xticks

plt.style.use('seaborn')

from ARPM_utils import save_plot
from CONFIG import GLOBAL_DB, TEMPORARY_DB
# -

# ## Upload database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_Stresserror'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_Stresserror'), squeeze_me=True)

k_ = db['k_']
M = db['M']
expectation = db['expectation']
er_rob_M = db['er_rob_M']
er_ens_M = db['er_ens_M']
# -

# ## Compute the error of the shrinkage estimator for different confidence levels

# +
gamma = arange(0,1.02,0.02)  # confidence levels

er_S = zeros((k_, len(gamma)))
for i in range(len(gamma)):
    S = gamma[i]*3 + (1-gamma[i])*M
    for k in range(k_):
        L_S = (S[:,k]-expectation[k]) ** 2
        er_S[k, i] = mean(L_S)
# -

# ## Compute robust and ensemble errors for each confidence level

# +
er_rob_S, i_S = npmax(er_S, axis=0) , argmax(er_S, axis=0) # robust errors

er_ens_S = mean(er_S, axis=0)  # ensemble errors
# -

# ## Find the optimal confidence level for both robust and ensemble approaches

# +
i_rob = argmin(er_rob_S)
i_ens = argmin(er_ens_S)

c_rob = gamma[i_rob]  # optimal confidence level for the robust approach
c_ens = gamma[i_ens]  # optimal confidence level for the ensemble approach
# -

# ## Create figures that compare the sample mean estimator and shrinkage estimator

# +
c0_bl = [0.27, 0.4, 0.9]

# axis settings
M_rob = er_rob_M
S_rob = er_rob_S[i_rob]
M_ens = er_ens_M
S_ens = er_ens_S[i_ens]
Y_rob = (M_rob + S_rob) / 2
dY_rob = abs((M_rob - Y_rob))
Y_ens = (M_ens + S_ens) / 2
dY_ens = abs((M_ens - Y_ens))

# robust error
figure()
bar([1, 2], [M_rob, S_rob], facecolor=c0_bl)
xlim([0.3, 2.6])
ylim([Y_rob - 1.9*dY_rob, Y_rob + 1.5*dY_rob])
xticks([1, 2],['Sample mean','Shrinkage'])
title('Robust errors')
con1 = 'Optimal shrinkage level: c = %.2f' %c_rob
plt.text(1, Y_rob + 1.3*dY_rob, con1)

# ensemble error
figure()
bar([1, 2], [M_ens, S_ens], facecolor=c0_bl)
xlim([0.3, 2.6])
ylim([Y_ens - 1.9*dY_ens, Y_ens + 1.5*dY_ens])
xticks([1, 2],['Sample mean','Shrinkage'])
title('Ensemble errors')
con1 = 'Optimal shrinkage level: c = %.2f'% c_ens
plt.text(1, Y_ens + 1.3*dY_ens, con1);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
