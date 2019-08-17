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

# # s_spectral_representation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_spectral_representation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-cross-spectr-propp-copy-5).

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import toeplitz

from arpym.statistics import simulate_normal, simulate_var1
from arpym.tools import pca_cov, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-parameters)

t_ = 10**3  # length of process
j_ = 10  # number of simulations of paths
b = 0.6  # autoregression parameter of the AR(1)
mu_eps = 0  # expectation of shocks in AR(1)
sigma2_eps = 1-b**2  # variance of shocks in AR(1)
t = 300  # lags

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-implementation-step01): Simulation of AR(1)

# +
if b == 0:
    b = 10**-6

mu_x = mu_eps/(1-b)  # expectation of (stationary) AR(1)
sigma2_x = sigma2_eps/(1-b**2)  # variance of (stationary) AR(1)

x = np.zeros((t_, j_))
for j in range(j_):
    x0 = simulate_normal(mu_x, sigma2_x, 1)
    x[:, j] = simulate_var1(x0,
                            np.atleast_2d(b),
                            np.atleast_2d(mu_eps),
                            np.atleast_2d(sigma2_eps),
                            t_-1,
                            j_=1).squeeze()
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-implementation-step02): Choose subsequent observations

t0 = np.int(t_/2)  # choose far from initial point to have more stationarity
x_vec = x[t0-t:t0+t+1, :]
mu_x_vec = mu_x*np.ones((2*t+1, j_))
t_vec = np.linspace(-t, t, 2*t+1)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-implementation-step03): Compute covariance matrix of random vector x_vec

k_x = b**(np.arange(2*t+1))/(1-b**2)*sigma2_eps  # autocovariance fun
cv_x = toeplitz(k_x)  # autocovariance matrix

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-implementation-step04): Compute eigenvectors/eigenvalues

# +
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

delta_omega = 2*np.pi/(2*t+1)
eta = np.sqrt(np.pi/(delta_omega))*e
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-implementation-step05): Compute spectral density of the AR(1)

# +
# frequences
omega_vec = np.zeros((2*t+1, 1))
for j in range(1, 2*t+2):
    omega_vec[j-1] = ((-1)**(j-1))*j*np.pi/(2*t+1)
omega_vec = np.sort(omega_vec, axis=None)

ktilde_x = sigma2_eps/(1-2*b*np.cos(omega_vec)+b**2)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-implementation-step06): Compute principal factors

z_pc_omega = e.T@(x_vec - mu_x_vec)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-implementation-step07): Compute rescaled principal factors

delta_y_omega = np.sqrt(delta_omega)*z_pc_omega

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_representation-implementation-step08): Compute the orthogonal increments process

y_omega = np.cumsum(delta_y_omega, 0)  # cumulative variable

# ## Plots

# +
plt.style.use('arpm')
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
lightgrey = [.8, .8, .8]
darkgrey = [.1, .1, .1]
darkgreen = [0, 0.7, 0]
lightred = [.9, .6, .6]
darkred = [.9, 0, 0]
lightblue = [181/255, 209/225, 223/225]
omegalim = [-np.pi, np.pi]
taulim = [t_vec[0], t_vec[-1]]

gs0 = gridspec.GridSpec(2, 2)

ax1 = plt.Subplot(f, gs0[0, 0])
f.add_subplot(ax1)
ax1.tick_params(labelsize=14)
for j in range(1, j_):
    plt.plot(t_vec, np.squeeze(x_vec[:, j]), color=lightgrey, linewidth=0.2)
plt.ylabel('$X_t$', fontsize=17)
plt.xlabel('$t$', fontsize=17)
p1 = plt.plot(t_vec, np.squeeze(x_vec[:, j]), color='k', linewidth=1)
plt.title('AR(1) process, b = ' + str(b), fontsize=20)

ax2 = plt.Subplot(f, gs0[0, 1])
f.add_subplot(ax2)
ax2.tick_params(labelsize=14)
for j in range(1, j_):
    plt.plot(omega_vec, np.real(np.squeeze(y_omega[:, j])), color=lightgrey,
             linewidth=0.2)
plt.plot(omega_vec, np.real(np.squeeze(y_omega[:, 0])), color=darkgrey,
         linewidth=1, label='Orth. incr. process')
plt.plot(omega_vec, 2*np.sqrt(np.cumsum(lambda2*delta_omega)),
         color=darkgreen, linewidth=0.9, label='2std')
plt.plot(omega_vec, -2*np.sqrt(np.cumsum(lambda2*delta_omega)),
         color=darkgreen, linewidth=0.9)
plt.legend()
plt.title('Orthogonal increments process', fontsize=20)
plt.xlabel('$\omega$', fontsize=17)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

ax3 = plt.Subplot(f, gs0[1, 0])
f.add_subplot(ax3)
ax3.tick_params(labelsize=14)
plt.plot(t_vec, eta[:, t], 'b', linewidth=2, label=r'$\eta_{\omega_1}$')
plt.plot(t_vec, eta[:, t+1], 'm', linewidth=2, label=r'$\eta_{\omega_3}$')
plt.plot(t_vec, eta[:, t-1], 'y', linewidth=2, label=r'$\eta_{\omega_{2}}$')
plt.plot(t_vec, eta[:, t+2], 'c', linewidth=2, label=r'$\eta_{\omega_5}$')
plt.plot(t_vec, eta[:, t-2], 'g', linewidth=2, label=r'$\eta_{\omega_{4}}$')
plt.legend()
plt.xlabel(r'$\omega$', fontsize=17)
plt.xticks([-300, -150, 0, 150, 300],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.title('Rescaled eigenvectors', fontsize=20)

ax4 = plt.Subplot(f, gs0[1, 1])
f.add_subplot(ax4)
ax4.tick_params(labelsize=14)
plt.bar(t_vec, lambda2, color='lightblue', label=r'$\lambda^2_\omega$')
plt.plot(t_vec, ktilde_x, color=darkred, linewidth=0.4, label=r'$\tilde{k}_X(\omega)$')
plt.legend()
plt.xticks([-300, -150, 0, 150, 300],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.xlabel(r'$\omega$', fontsize=17)
plt.title('Spectrum', fontsize=20)

add_logo(f, location=4)
plt.tight_layout()
