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

# # S_FactorReplicationTest [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FactorReplicationTest&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-cross-sec-reg-num-test).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import ones, zeros, cov, eye, r_
from numpy.linalg import solve, pinv
from numpy.random import randn

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, xlim, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from NormalScenarios import NormalScenarios
from MultivRsquare import MultivRsquare

# input parameters
n_ = 100  # max dimension of target X
nstep = range(10,n_+1)  # target dimension steps
j_ = 1000  # number of simulations
k_ = 5  # dimension of factors Z
sigma2_Z = eye(k_)  # factor covariance
sig2_U = 0.8

stepsize = len(nstep)
R2_Reg = zeros((stepsize, 1))
R2_CS = zeros((stepsize, 1))

R2_XReg = zeros((stepsize, 1))
R2_XCS = zeros((stepsize, 1))

for n in range(stepsize):

    # ## Generate a sample from the joint distribution of the factors and residuals

    mu_ZU = zeros((k_ + nstep[n], 1))  # expectation
    sig2_ZU = zeros((k_, nstep[n]))  # systematic condition
    d = sig2_U*ones((nstep[n], 1))
    sigma2_U = np.diagflat(d * d)  # idiosyncratic condition
    sigma2_ZU = r_[r_['-1',sigma2_Z, sig2_ZU], r_['-1',sig2_ZU.T, sigma2_U]]  # covariance

    Z_U,_ = NormalScenarios(mu_ZU, sigma2_ZU, j_)  # joint sample
    # Z_U = Z_U.T  # ensure Z_U is (k_ + n_) x nsim

    # ## Generate target sample according to systematic-idiosyncratic LFM

    Z = Z_U[:k_,:]  # observable factors sample
    U = Z_U[k_:,:]  # observable residuals sample
    beta_XZ = randn(nstep[n], k_)  # observable loadings

    i_n = eye(nstep[n])
    X = r_['-1',beta_XZ, i_n]@Z_U  # target sample
    sigma2_X = beta_XZ@sigma2_Z@beta_XZ.T + sigma2_U  # (low-rank diagonal) covariance

    sigma2_XZ = beta_XZ@sigma2_Z  # covariance of target and factors

    invres2 = np.diagflat(1 / (d * d))  # inverse residuals covariance
    inv_sig2 = invres2-(invres2@beta_XZ).dot(pinv(beta_XZ.T@invres2@beta_XZ
          + solve(sigma2_Z,eye(sigma2_Z.shape[0]))))@beta_XZ.T@invres2  # inverse residuals covariance

    # ## Recovered regression factors

    beta_Reg = (sigma2_XZ.T)@inv_sig2  # regression loadings of Z over X
    Z_Reg = beta_Reg@X  # regression recovered factor sample

    # ## Recovered cross-sectional factors

    beta_fa = beta_XZ
    invres2_fa = invres2
    beta_CS = solve(beta_fa.T@invres2_fa@beta_fa,beta_fa.T@invres2_fa)  # pseudo inverse
    Z_CS = beta_CS@X  # cross-sectional extracted factor sample

    # ## Recover X via regression of X over Z

    beta_XZReg = sigma2_XZ@solve(sigma2_Z,eye(sigma2_Z.shape[0]))  # regression loadings of X over Z
    X_Reg = beta_XZReg@Z  # regression recovered target

    # ## Compute X via cross-sectional on Z

    gamma = solve(beta_XZ.T@invres2@beta_XZ + solve(sigma2_Z,eye(sigma2_Z.shape[0])),beta_XZ.T)@invres2
    X_CS = beta_XZ@gamma@X

    # ## Compute the r-square at dimension nstep[n]

    R2_Reg[n] = MultivRsquare(cov(Z_Reg-Z), sigma2_Z, eye(k_))
    R2_CS[n] = MultivRsquare(cov(Z_CS-Z), sigma2_Z, eye(k_))
    R2_XReg[n] = MultivRsquare(cov(X_Reg-X), sigma2_X, sigma2_X)
    R2_XCS[n] = MultivRsquare(cov(X_CS-X), sigma2_X, sigma2_X)
# -

# ## Scatter plot Z vs factor replications

figure()
scatter(Z_Reg[0], Z[0], marker='*')
scatter(Z_CS[0], Z[0], marker='o',facecolors='none', color=[1, 0.5, 0])
xlabel('Recovered Factors')
ylabel('Z')
title('Scatter plot for n = %d and k = %d' % (n_,k_))
legend(['Regression Z', 'Cross-Sec Z']);
plt.tight_layout()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# ## Plot the r-squares for each target dimension

figure()
plot(nstep, R2_Reg, 'r', linewidth=1.2)
plot(nstep, R2_CS, 'g', linewidth=1.2)
plot(nstep, ones(stepsize), 'b', linewidth=2, )
xlabel(r'$n_{1}$')
ylabel(r'$R^{2}$')
xlim([min(nstep),max(nstep)])
legend(['Regression $R^2$', 'Cross-Sec $R^2$']);
plt.tight_layout()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# ## Scatter plot X vs factor replications

figure()
scatter(X_Reg[0], X[0], marker='*')
scatter(X_CS[0], X[0], marker='o', facecolors='none', color=[1, 0.5, 0])
xlabel('Recovered Target')
ylabel('X')
title('Scatter plot for n = %d and k = %d' % (n_,k_))
legend(['Regression X', 'Cross-Sec X']);
plt.tight_layout()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# ## Plot the r-squares for each market dimension

figure()
plot(nstep, R2_XReg, 'r', linewidth=1.2)
plot(nstep, R2_XCS, 'g', linewidth=1.2)
plot(nstep, ones(stepsize), 'b', lw=2)
xlabel('n')
ylabel(r'$R^{2}$')
xlim([min(nstep),max(nstep)])
legend(['Regression $R^2$', 'Cross-Sec $R^2$']);
plt.tight_layout()
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

