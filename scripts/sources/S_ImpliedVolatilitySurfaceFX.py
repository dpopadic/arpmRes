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

# # S_ImpliedVolatilitySurfaceFX [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ImpliedVolatilitySurfaceFX&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerImplVolFX).

# ## Prepare the environment

# +
import sys, os.path as path, os

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import zeros, ceil, linspace, sqrt, tile, arange
from numpy import min as npmin, max as npmax

from scipy.stats import norm
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, ylim, subplots, ylabel, \
    xlabel, title, xticks, scatter
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')
# -

# ## Upload data from db_ImpliedVol_FX

# +
from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, struct_to_dict
from intersect_matlab import intersect

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_ImpliedVol_FX'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_ImpliedVol_Fx'), squeeze_me=True)

db_ImpliedVol_FX = struct_to_dict(db['db_ImpliedVol_FX'])

# implied volatility surface for GBPUSD rate (in percentage format)

tau = db_ImpliedVol_FX.TimesToMaturity
delta =  db_ImpliedVol_FX.Delta
sigma_delta  =  db_ImpliedVol_FX.Sigma
t_ = sigma_delta.shape[2]
n_ = len(delta)
# -

# ## Plot the implied volatility surface and the evolution of implied volatility for the desired values of delta-moneyness and times to maturity

# +
_,tauIndex,_ = intersect(tau,1) # select 1 year of maturity
meanIndex_delta = int(ceil((n_)/2))-1

x,y = np.meshgrid(delta,tau)

f,ax = subplots(1,1,subplot_kw={'projection':'3d'})
ax.view_init(30,-120)
ax.plot_surface(x,y,sigma_delta[:,:,t_-1])
ax.scatter(x.flatten(),y.flatten(),sigma_delta[:,:,t_-1].flatten(),edgecolor='k')
plot(delta[[0]],tau[tauIndex],sigma_delta[tauIndex,0,t_-1],marker='.', color='r',markersize=20)
plot(delta[[meanIndex_delta]],tau[tauIndex],sigma_delta[tauIndex,meanIndex_delta,t_-1],marker='.', color='b'
     ,markersize=20)
plot(delta[[n_-1]],tau[tauIndex],sigma_delta[tauIndex,n_-1,t_-1],marker='.', color='g',markersize=20)
xlabel('$\delta$-moneyness', labelpad=10)
ylabel('Time to maturity (years)', labelpad=10)
ax.set_zlabel('Volatility (%)')
xlim([min(delta), max(delta)])
ylim([min(tau), max(tau)])
xticks(delta)
title('Implied volatility surface SP500');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

figure()
plot(arange(t_),sigma_delta[tauIndex,0,:].flatten(),'r')
plot(arange(t_),sigma_delta[tauIndex,meanIndex_delta,:].flatten(),'b')
plot(arange(t_),sigma_delta[tauIndex,n_-1,:].flatten(),'g')
xlim([1, t_])
xlabel('Time')
ylabel('Volatility (%)')
legend(['$\delta$=0.10','$\delta$=0.50','$\delta$=0.90'])
title('Imp. vol. evol.: 1 year to maturity')
plt.grid(True);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
