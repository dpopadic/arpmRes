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

# # S_FacRepNormTest [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_FacRepNormTest&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-fac-rep-port-norm).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

import numpy as np
from numpy import arange, array, ones, zeros, diag, eye, tile, r_
from numpy.linalg import solve
from numpy.random import rand
from numpy.random import multivariate_normal as mvnrnd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, legend, scatter, ylabel, \
    xlabel, title

plt.style.use('seaborn')

from ARPM_utils import save_plot
from MultivRsquare import MultivRsquare

# input parameters
n_ = 500  # max market dimension
nstep = arange(10, n_+25,25)  # market dimension steps
j_ = 1000  # number of simulations
k_ = 1  # number of factors
sig2_Z_ = 1  # factor variance
r = 0.02  # risk-free rate

stepsize = len(nstep)
R2 = zeros((stepsize, 1))
for n in range(stepsize):

    # ## Generate a sample from the joint distribution of the factor and the residuals

    mu_Z_U = zeros((k_ + nstep[n], 1))  # expectation
    sig_Z_U = zeros((k_, nstep[n]))  # systematic condition
    d = rand(nstep[n], 1)  # residuals standard deviations
    sig2_U = np.diagflat(d * d)  # idiosyncratic condition
    sig2_Z_U = r_[r_['-1',array([[sig2_Z_]]), sig_Z_U], r_['-1',sig_Z_U.T, sig2_U]]  # covariance

    Z_U = mvnrnd(mu_Z_U.flatten(), sig2_Z_U, j_)
    Z_U = Z_U.T  # ensure Z_U is n_ x nsim

    Z_ = Z_U[0]  # factor sample

    # ## Compute the P&L's: P = alpha + beta@Z_ + U

    alpha = rand(nstep[n], 1)  # shift parameter (P&L's expectation)
    beta = rand(nstep[n], k_)  # loadings
    i_n = eye(nstep[n])
    P = tile(alpha, (1, j_)) + r_['-1',beta, i_n]@Z_U  # sample
    sig2_P = beta@array([[sig2_Z_]])@beta.T + sig2_U  # (low-rank diagonal) covariance

    # ## Compute the sample of the factor-replicating portfolio

    s2 = i_n
    betap = solve(beta.T@s2@beta,beta.T@s2)  # pseudo inverse of beta
    P_Z = betap@P  # sample
    mu_P_Z = betap@alpha  # expectation
    sig2_P_Z = betap@sig2_P@betap.T  # covariance

    # ## Compute premium via APT

    v = ones((nstep[n], 1))  # current values of P&L's
    lam = betap@(alpha - r*v)
    Z = Z_ + lam  # shifted factors

    # ## Compute the r-square at dimension nstep[n]

    sig2_U_Z_ = betap@sig2_U@betap.T  # covariance of P_Z - r@ betap@v - lam - Z_
    sigvec_Z_ = diag(array([sig2_Z_]))
    R2[n] = MultivRsquare(sig2_U_Z_, array([[sig2_Z_]]), np.diagflat(1 / sigvec_Z_))
# -

# ## Scatter plot of factor plus premium vs factor replicating portfolios P&L's in excess of the risk-free investement

figure()
scatter(Z, P_Z - r*betap@v, marker='.',s=0.5)
scatter(lam, mu_P_Z - r*betap@v, marker='.', color='r', s=50)
xlabel('Z')
ylabel('Excess PL factor replicating portfolio')
title('Scatter plot for n = %d' % n_)
legend(['sample', 'expectation']);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

# ## Plot the r-squares for each market dimension

# +
figure()

plot(nstep, R2, 'r', lw=1.2)
plot(nstep, ones(stepsize), 'b', lw=2)
xlabel('n')
ylabel('r-square')
title('Factor-replicating portfolio convergence');
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
