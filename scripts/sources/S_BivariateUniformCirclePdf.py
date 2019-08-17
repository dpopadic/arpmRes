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

# # S_BivariateUniformCirclePdf [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_BivariateUniformCirclePdf&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExUnifCircleBivariate).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import array, ones, pi, linspace, arange, r_
from numpy import min as npmin, max as npmax

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlim, ylim, subplots, xticks, yticks
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d as a3d

plt.style.use('seaborn')

from ARPM_utils import save_plot
from SmoothStep import SmoothStep

# parameters
Radius = 1
Condition_X1 = array([.45, .9])*Radius
# -

# ## Compute the pdf of the uniform distribution

# +
Refine = 200

Min = -1.02*Radius
Max = 1.02*Radius
Range_X1 = linspace(Min,Max,Refine)
Range_X2 = Range_X1
Hight = 1 / (pi*Radius ** 2)
Min = 0
Max = 1.2*Hight
Range_Z = linspace(Min,Max,Refine)

X1, X2 = np.meshgrid(Range_X1, Range_X2)
Epsilon = .02
pdf_X = Hight*SmoothStep(Radius ** 2 - (X1 ** 2 + X2 ** 2), Epsilon)
# -

# ## Plot the uniform pdf and the conditional pdf

# +
f, ax = subplots(1,1,subplot_kw={'projection':'3d'})

ax.plot_surface(X1, X2, pdf_X,color='lightgray')
ax.view_init(30,-126)
m = npmin(pdf_X)
M = npmax(pdf_X)

uu = (pdf_X - m) / (M - m)

for i in range(len(Condition_X1)):
    Z_range =[0, npmax(pdf_X)*1.05]
    Xx =array([Range_X1[0], Range_X1[0], Range_X1[-1], Range_X1[-1]])
    Yy = Condition_X1[i]*ones(len(Xx))
    Zz =array([Z_range[0], Z_range[-1], Z_range[-1], Z_range[0]])
    xticks(arange(-1,1.5,0.5))
    yticks(arange(-1,1.5,0.5))
    vtx = [list(zip(Xx, Yy, Zz))]
    tri =a3d.Poly3DCollection(vtx,linewidth=1.5,zorder=0)
    tri.set_facecolor('white')
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)

xlim([Range_X1[0], Range_X1[-1]])
ylim([Range_X2[0], Range_X2[-1]]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
