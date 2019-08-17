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

# # s_bandpass_filter_ar1 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bandpass_filter_ar1&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-cross-spectr-propp-copy-6).

# +
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

from arpym.statistics import simulate_normal, simulate_var1
from arpym.tools.logo import add_logo
# -

# ##  Input parameters

b = 0.7  # autoregression parameter
mu_eps = 0  # location of the shocks
sigma2_eps = 1-b**2  # dispersion of the shocks
t_ = 350  # lags
t_vec = np.arange(2*t_+1)
tau = 100  # truncation
tau_vec = np.arange(-tau, tau+1)
omega0 = 1/4*np.pi
omega1 = 1/2*np.pi

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bandpass_filter_ar1-implementation-step01): Simulate stationary AR(1) process

mu_x = mu_eps/(1-b)  # expectation of (stationary) AR(1)
sigma2_x = sigma2_eps/(1-b**2)  # variance of (stationary) AR(1)
x0 = simulate_normal(mu_x, sigma2_x, 1)
x = simulate_var1(x0, np.atleast_2d(b), np.atleast_2d(mu_eps),
                            np.atleast_2d(sigma2_eps),
                            2*t_, j_=1).squeeze()

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_bandpass_filter_ar1-implementation-step02): Compute spectral density and bandpass filter

# +
omega_vec = np.zeros((2*t_+1, 1))
for j in range(1, 2*t_+2):
    omega_vec[j-1] = ((-1)**(j-1))*j*np.pi/(2*t_+1)
omega_vec = np.sort(omega_vec, axis=None)
ktilde_x = sigma2_eps / (1 - 2*b*np.cos(omega_vec) + b**2)

# preliminary computations
h_tilde = np.zeros(len(omega_vec))
int_ktilde_x_plus, _ = quad(lambda omega: sigma2_eps / (1 - 2*b*np.cos(omega) + b**2),
                       omega0, omega1)
int_ktilde_x_minus, _ = quad(lambda omega: sigma2_eps / (1 - 2*b*np.cos(omega) + b**2),
                       -omega1, -omega0)
int_ktilde_x = int_ktilde_x_plus + int_ktilde_x_minus

resc_h_tilde = np.sqrt(sigma2_x/((1/(2*np.pi))*int_ktilde_x))

# compute h_tilde
for omega in range(len(omega_vec)):
    if np.abs(omega_vec[omega]) >= omega0 and \
                                            np.abs(omega_vec[omega]) <= omega1:
        h_tilde[omega] = resc_h_tilde
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_bandpass_filter_ar1-implementation-step03): Compute bandpass impulse response and approx. bandpass filter

# +
# compute bandpass impulse response
h = np.zeros(len(tau_vec))

for tau in range(len(tau_vec)):
    int_cos, _ = quad(lambda omega: np.cos(omega*tau_vec[tau]),
                      omega0, omega1)
    h[tau] = np.sqrt(sigma2_x/((np.pi/2)*int_ktilde_x))*int_cos

# approximated h_tilde
h_tilde_approx = np.zeros(len(omega_vec), dtype=complex)
for omega in range(len(omega_vec)):
    h_tilde_approx[omega] = np.sum(np.exp(-1j *
                                          omega_vec[omega]*tau_vec[:])*h[:])
h_tilde_approx = np.real(h_tilde_approx)

# update times
t_vec = t_vec[1+tau:-1-tau]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_bandpass_filter_ar1-implementation-step04): Compute filtered process

y = np.zeros(len(t_vec))
for t in range(len(t_vec)):
    fil_proc = 0
    for tau in range(len(tau_vec)):
        fil_proc = fil_proc + h[tau]*x[t_vec[t]-tau_vec[tau]]
    y[t] = np.real(fil_proc)

# ##  Plots

# +
plt.style.use('arpm')

fig, ax = plt.subplots(3, 1)
lightblue = [.4, .7, 1]

# process
plt.sca(ax[0])
plt.plot(t_vec, x[t_vec], color='k', linewidth=0.5)
plt.plot(t_vec, y, color=lightblue, linewidth=0.8)
plt.xlabel('$t$')
plt.ylabel('$X_t$')
plt.legend(['Process', 'Filtered process'])
strplot = '$\Omega$ = [%.2f, %.2f]' % (omega0, omega1)
plt.title('Filtering, ' + strplot)

# spectral density
plt.sca(ax[1])
p1 = plt.plot(omega_vec, ktilde_x, color='k', linewidth=0.8,
              label='Spectral density')
plt.twinx()
p2 = plt.plot(omega_vec, h_tilde, '--', color=lightblue,
              label='Bandpass filter')
p3 = plt.plot(omega_vec, h_tilde_approx, '-', color=lightblue, linewidth=0.8,
              label='Approx. bandpass filter')
plt.tick_params(axis='y', colors=lightblue)
plt.xlabel('$\omega$')
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
plt.grid(False)
lns = p1 + p2 + p3
labs = [l.get_label() for l in lns]
ax[1].legend(lns, labs, loc=0)

# impulse response
plt.sca(ax[2])
plt.bar(tau_vec, h, facecolor=lightblue, edgecolor=lightblue)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$h_{\tau}$')
plt.legend(['Bandpass impulse response'])

add_logo(fig)
plt.tight_layout()
