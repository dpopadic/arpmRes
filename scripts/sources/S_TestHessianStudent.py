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

# # S_TestHessianStudent [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_TestHessianStudent&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-mfpellipt-copy-2).

# ## Prepare the environment
import os
# # +
import os.path as path
import sys

from scipy.io import loadmat

import statsmodels.sandbox.distributions.mv_normal as mvd

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import arange, reshape, sign, where, diag, log, exp, sqrt, tile, diagflat, r_, zeros, newaxis, array
from numpy import sum as npsum
from numpy.linalg import eig
from numpy.random import randn, np

import matplotlib.pyplot as plt

from ARPM_utils import multivariate_t_rvs as mvtrvs, multivariate_t_distribution as mvtpdf
from numHess import numHess

plt.style.use('seaborn')

from MaxLikelihoodFPLocDispT import MaxLikelihoodFPLocDispT
# -

# ## Set the initial parameters and generate the dataset

# +
nu = 10  # degrees of freedom
t_ = 2000  # dataset's len
i_ = 1  # dimension of the t-Student random variable
mu = 2*randn(i_, 1)  # mean vector of dimension (i_ x 1)
sigma_temp = 2*randn(i_, i_)
sigma2 = sigma_temp.T@sigma_temp  # covariance matrix of dimension (i_ x i_)

epsi_temp = mvtrvs(zeros(sigma2.shape[0]), sigma2, nu, t_).T

epsi = diagflat(sqrt(diag(sigma2)))@epsi_temp + tile(mu, (1, t_))  # dataset of dimension (i_ x t_end) from a t() distribution
# -

# ## Set the Flexible Probability profile for MLFP estimation (exponential decay with half life 12 months)

lam = log(2) / 360
p = exp(-lam*arange(t_, 1 + -1, -1)).reshape(1,-1)
p = p /npsum(p)

# ## Compute MLFP estimators of location and dispersion from the sample

mu_MLFP, sigma2_MLFP, err1 = MaxLikelihoodFPLocDispT(epsi, p, nu, 10**-15, 1)

# ## Define the likelihood function

# +
mvt = mvd.MVT(array([0]),array([[1]]),df=nu)
mvtpdf = mvt.pdf

likelihood = lambda theta: npsum(p * np.real(log((mvtpdf((epsi - tile(theta[:i_], (1, t_))).T@diagflat(
    1 / sqrt(reshape(theta[i_:i_*(1 + i_)], (i_, -1),'F').astype(np.complex128))))).astype(np.complex128).T)))
# -

# ## Compute the Hessian matrix

hessian, err2 = numHess(likelihood, r_[mu_MLFP[...,newaxis],sigma2_MLFP])

# ## Compute the eigenvalues of the Hessian matrix

# +
Diag_lambda2, e = eig(hessian)
lambda2 = Diag_lambda2

answer = where(sign(lambda2) > -1)  # this array should be empty
