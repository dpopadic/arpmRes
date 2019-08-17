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

# # s_autocov_spec_dens_ar1 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_autocov_spec_dens_ar1&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-autocov-ar-copy-1).

# +
import numpy as np
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_autocov_spec_dens_ar1-parameters)

b = 0.6   # autoregression parameter of AR(1)
sigma2_eps = 1 - b ** 2   # variance of shocks in AR(1)
t = 30   # lags


# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_autocov_spec_dens_ar1-implementation-step01): Compute autocovariance function

tau_vec = np.arange(-t, t)
k_x = sigma2_eps * (b ** abs((tau_vec)) / (1 - b ** 2))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_autocov_spec_dens_ar1-implementation-step02): Compute spectral density

omega_vec = np.zeros((2*t+1, 1))
for j in range(1, 2*t+2):
    omega_vec[j-1] = ((-1)**(j-1))*j*np.pi/(2*t+1)
omega_vec = np.sort(omega_vec, axis=None)
ktilde_x = sigma2_eps / (1 - 2 * b * np.cos(omega_vec) + b ** 2)

# ## Plots

# +
plt.style.use('arpm')
darkred = [.9, 0, 0]
lightgrey = [.8, .8, .8]

plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
tau_vec = np.arange(-t, t)

gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0, :])
ax1.bar(tau_vec, k_x, color=lightgrey)
ax1.set_xlabel(r'$\Delta t$')
ax1.set_xlim([tau_vec[0], tau_vec[-1]])
ax1.set_ylabel(r'$k_X(\Delta t)$')
ax1.set_title('Autocovariance')

ax2 = plt.subplot(gs[1, :])
ax2.plot(omega_vec, ktilde_x, lw=1, color=darkred)
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xlabel(r'$\omega$')
ax2.set_ylabel(r'$\tilde{k}_X(\omega)$')
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax2.set_title('Spectral density')

add_logo(f, location=4)
plt.tight_layout()
