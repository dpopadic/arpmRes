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

# # S_LinTransEuclidBasis [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_LinTransEuclidBasis&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBEuclidBasAffiTransf).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, diag, eye, sqrt, arange

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, quiver, xticks
np.seterr(invalid='ignore')

plt.style.use('seaborn')

from ARPM_utils import save_plot
from Riccati import Riccati

# input parameters
s2X = array([[5, -7], [- 7, 10]])
b = array([[2.5, 1], [0, 3]])
# -

# ## Compute a Riccati root of the correlation and the vectors of X

# +
svecX = diag(sqrt(s2X))
c2X = np.diagflat(1 / svecX)@s2X@np.diagflat(1 / svecX)
cX = Riccati(eye(2), c2X)

xp = np.diagflat(svecX)@cX
# -

# ## Compute the vectors of Z via linear transformation

zp = b@xp

# ## Compute the covariance matrix of Z by means of the affine equivariance

s2Z = b@s2X@b.T

# ## Compute a the Riccati root of the correlation and the vectors of Z

# +
svecZ = diag(sqrt(s2Z))
c2Z = np.diagflat(1 / svecZ)@s2Z@np.diagflat(1 / svecZ)

cZ = Riccati(eye(2), c2Z)
zzp = np.diagflat(svecZ)@cZ
# -

# ## Display the Euclidean vectors

# +
figure()

quiver(0, 0, zp[0, 0], zp[1, 0], color = 'm', lw= 1, angles='xy',scale_units='xy',scale=1)
quiver(0, 0, zzp[0, 0], zzp[1, 0], color = 'b', lw= 1, angles='xy',scale_units='xy',scale=1)
quiver(0, 0, zp[0, 1], zp[1, 1], color = 'm', lw= 2, angles='xy',scale_units='xy',scale=1)
quiver(0, 0, zzp[0, 1], zzp[1, 1], color = 'b', lw= 2, angles='xy',scale_units='xy',scale=1)
quiv1 = plot(0, 0, color='b', lw= 0, marker='.')
quiv2 = plot(0, 0, color='m', lw= 0, marker='.')
xticks(arange(-3,3.5,0.5))
plot(0, 0, 'o',markeredgecolor='k',markerfacecolor='w')
plt.grid(True)
plt.axis([-3,3,-8,8])
legend(handles=[quiv1[0],quiv2[0]],labels=[r'{$z_1$, $z_2$}', '{$zz_1$, $zz_2$}']);  # legend for quiver plots not supported yet
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

