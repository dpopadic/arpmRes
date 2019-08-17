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

# # s_toeplitz_spectral [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_toeplitz_spectral&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_toeplitz_spectral).

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from arpym.tools import pca_cov, add_logo

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_toeplitz_spectral-parameters)

b = 0.5
sigma2_eps = 1-b**2
tvec = range(200, 600)
delta_orth = np.zeros((len(tvec), 1))
delta_decomp = np.zeros((len(tvec), 1))
delta_spectrum = np.zeros((len(tvec), 1))

for t in range(1, len(tvec)+1):

    # ## Step 1: Compute Autocovariance function

    k_x = sigma2_eps*b**(np.arange(2*t+1))/(1-b**2)  # autocovariance fun
    cv_x = toeplitz(k_x)  # autocovariance matrix

    # ## Step 2: Compute theoretical eigenvectors
    omega_vec = np.zeros((2*t+1, 1))
    for j in range(1, 2*t+2):
        omega_vec[j-1] = ((-1)**(j-1))*j*np.pi/(2*t+1)
    omega_vec = np.sort(omega_vec, axis=None)
    delta_omega = 2*np.pi/(2*t+1)
    s = np.zeros((2*t+1, t))
    c = np.zeros((2*t+1, t+1))
    for j in range(t):
        s[:, j] = np.sin(omega_vec[j]*np.linspace(-t, t, 2*t+1))
    for j in range(t+1):
        c[:, j] = np.cos(omega_vec[j+t]*np.linspace(-t, t, 2*t+1))
    p = np.c_[s, c]

    # ## Step 3: Compute spectral density of the AR(1)

    ktilde_x = sigma2_eps/(1-2*b*np.cos(omega_vec) + b**2)

    # ## Step 4: Compute empirical eigenvelues and eigenvectors

    e, lambda2 = pca_cov(cv_x)
    if b < 0:
        ind_asc = np.argsort(lambda2)
        lambda2 = lambda2[ind_asc]
        e = e[:, ind_asc]
    lambda2_new = []
    ind_e = []*(2*t+1)
    for n in range(1, 2*t+2):
        if n % 2 == 1:
            lambda2_new = np.append(lambda2_new, lambda2[n-1])
            ind_e = np.append(ind_e, n-1)
        else:
            lambda2_new = np.append(lambda2[n-1], lambda2_new)
            ind_e = np.append(n-1, ind_e)
    ind_e1 = [int(i) for i in ind_e]
    lambda2 = lambda2_new
    e = e[:, ind_e1]

    # ## Step 5: Compute spectrum error

    delta_spectrum[t-1] = linalg.norm(lambda2-ktilde_x)/linalg.norm(ktilde_x)

    # ## Step 6: Compute decomposition error

    cv_x_recov = p@np.diag(ktilde_x)@p.T
    eta = np.sqrt(np.pi/(delta_omega))*e
    delta_decomp[t-1] = linalg.norm(eta@np.diag(lambda2)@eta.T-cv_x_recov)/linalg.norm(cv_x_recov)

    # ## Step 7: Compute orthogonalization error

    delta_orth[t-1] = linalg.norm(p.T@p-np.pi/(delta_omega)*np.eye(2*t+1))/linalg.norm(np.pi/(delta_omega)*np.eye(2*t+1))

# ## Plots

# +
plt.style.use('arpm')
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
darkgreen = [0, 0.7, 0]
darkred = [.9, 0, 0]
darkgrey = [.1, .1, .1]

mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
taulim = [tvec[0], tvec[-1]]
plt.plot(tvec, delta_spectrum, color='darkgreen', linewidth=1)
plt.plot(tvec, delta_decomp, color='darkred', linewidth=1)
plt.plot(tvec, delta_orth, color='darkgrey', linewidth=1)
plt.xlabel('$t$', fontsize=17)
plt.legend([r'Spectrum error', r'Decomposition error', r'Orthogonalization error'])
plt.title('Spectral theorem for Toeplitz matrices', fontsize=20)
add_logo(f, location=4)
plt.tight_layout()
