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

# # s_dyn_principal_component_var [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_dyn_principal_component_var&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-dyn-pc-var).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from arpym.statistics import simulate_var1, simulate_normal, multi_r2
from arpym.tools import transpose_square_root, add_logo

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-parameters)

n_ = 2  # number of target variables
k_ = 1  # number of factors
t_ = int(1e4)  # length of VAR(1) process
j_ = int(1e2)  # number of scenarios
delta_omega = 1e-3
sigma2 = np.eye(n_)  # scale matrix

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step00): Setup parameters

# +
t_vec = np.arange(t_)
tau_vec = np.arange(-j_, j_+1)
omega_vec = np.arange(-np.pi, np.pi, delta_omega)
m_ = len(omega_vec)
gamma = (2 * np.random.rand(4) - 1) * 0.99
theta = gamma * np.pi / 2

b = np.array([[np.sin(theta[0]), 0],
               [np.sin(theta[3])*np.sin(theta[2]),
                np.sin(theta[3])*np.cos(theta[2])]])

mu_epsi = np.zeros(n_)
s_1 = np.cos(theta[0])
s_2 = np.cos(theta[3])
rho = np.sin(theta[1])
sigma2_epsi = np.array([[s_1**2, rho*s_1*s_2],
                        [rho*s_1*s_2, s_2**2]])
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step01): Simulate VAR(1) process

# +
mu_inf = np.linalg.solve(np.eye(n_) - b, mu_epsi)
sigma2_inf = np.linalg.solve(np.eye(n_**2) - np.kron(b, b),
                             sigma2.reshape(n_**2, 1)).reshape(n_, n_)
x_tnow = simulate_normal(mu_inf, sigma2_inf, 1).reshape(n_)

x = simulate_var1(x_tnow, b, mu_epsi, sigma2_epsi, t_, j_=1).squeeze()
mu_x = np.linalg.solve((np.eye(n_) - b), mu_epsi)
sigma2_x = np.linalg.solve(np.eye(n_ ** 2) - np.kron(b, b),
                           sigma2_epsi.reshape(n_ ** 2, 1)).reshape(n_, n_)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step02): Compute spectral density

# +
ktilde_x = np.zeros((m_, n_, n_), dtype=complex)

sigma_epsi = transpose_square_root(sigma2_epsi)
for m in range(m_):
    ktilde_x_temp = np.linalg.solve(np.eye(n_, dtype=complex) -
                               np.exp(-omega_vec[m]*1j) * b, sigma_epsi)
    ktilde_x[m, :, :] = ktilde_x_temp @ ktilde_x_temp.conj().T
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step03): Principal components decomposition

# +
lam, e = np.linalg.eigh(ktilde_x)
lam_k = lam[:, -k_:][:, ::-1]
e_k = e[:, :, -k_:][:, :, ::-1]

sigma = transpose_square_root(sigma2)

beta_tilde_f = np.einsum('ij,ljk->lik', sigma, e_k)
gamma_tilde_f = np.einsum('ijk,kl->ijl',
                          e_k.conj().transpose((0, 2, 1)),
                          np.linalg.inv(sigma))
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step04): Computation of the filter h

# +
h_tilde_f = np.einsum('ijk,ikl->ijl', beta_tilde_f, gamma_tilde_f)

coef = np.exp(1j * np.outer(tau_vec, omega_vec))
h_f = np.real(np.tensordot(coef, h_tilde_f, axes=(1, 0)) *
              delta_omega / (2 * np.pi))
gamma_f = np.tensordot(coef, gamma_tilde_f, axes=(1, 0)) * \
          delta_omega / (2 * np.pi)
alpha_f = (np.eye(n_) - np.sum(h_f, axis=0)) @ mu_x
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step05): Compute the spectral density of predicted process

ktilde_x_pc_bar = np.einsum('ijk,ilk->ijl',
                   np.einsum('ijk,ikl->ijl', h_tilde_f, ktilde_x), h_tilde_f.conj())

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step06): Compute the principal components predicted process

# +
t_vec_pc = t_vec[tau_vec[-1]:-tau_vec[-1]]
t_pc = t_vec_pc.shape[0]
x_pc_bar = np.zeros((t_pc, n_), dtype=complex)
z_pc = np.zeros((t_pc, k_), dtype=complex)

for t in range(t_pc):
    x_tau = x[t_vec_pc[t] + tau_vec, :][::-1, :]
    x_pc_bar[t, :] = np.einsum('ijk,ik->j', h_f, x_tau) + alpha_f
    z_pc[t, :] = np.einsum('ijk,ik->j', gamma_f, x_tau)

x_pc_bar = np.real(x_pc_bar)
z_pc = np.real(z_pc)
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step07): update times of original process x

x = x[t_vec_pc, :]

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_dyn_principal_component_var-implementation-step08): Compute r-squared

u = x - x_pc_bar
sigma2_u = np.einsum('ijk,ilk->ijl',
                     np.einsum('ijk,ikl->ijl', np.eye(n_) - h_tilde_f, ktilde_x),
                     (np.eye(n_) - h_tilde_f).conj())
sigma2_u = np.sum(np.real(sigma2_u), axis=0) * delta_omega / (2 * np.pi)
r_2 = multi_r2(sigma2_u, sigma2_x, sigma2)

# ## Plots

# +
plt.style.use('arpm')

t_plot = t_vec_pc[1:150]
xlim = [t_plot[0], t_plot[-1]]
ylim = [-4, 4]

fig1, axes = plt.subplots(1, 2)
axes[0].plot(t_plot, x[t_plot, 0], 'b')
axes[0].plot(t_plot, x[t_plot, 0], 'r--')
axes[0].set_xlabel('$t$')
axes[0].set_ylabel('$x_1$')
axes[0].set_xlim(xlim)
axes[0].set_ylim(ylim)
axes[0].legend(['Process', 'Predicted process'])

axes[1].plot(t_plot, x[t_plot, 1], 'b')
axes[1].plot(t_plot, x[t_plot, 1], 'r--')
axes[1].set_xlabel('$t$')
axes[1].set_ylabel('$x_2$')
axes[1].set_xlim(xlim)
axes[1].set_ylim(ylim)
axes[1].legend(['Process', 'Predicted process'])
add_logo(fig1, size_frac_x=1/8)
plt.tight_layout()

fig2 = plt.figure()
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 1])
ax0 = plt.subplot(gs[0])
ax0.plot(ylim, ylim, 'k')
ax0.plot(x[t_plot, 0], x_pc_bar[t_plot, 0], 'r.')
ax0.set_xlabel('$x_1$')
ax0.set_ylabel('$\overline{x}_{1}^{pc}$')
ax0.set_xlim(ylim)
ax0.set_ylim(ylim)

ax1 = plt.subplot(gs[1])
ax1.plot(t_plot, z_pc[t_plot, 0], 'b')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$Z^{pc}$')
ax1.set_xlim(xlim)

ax2 = plt.subplot(gs[2])
ax2.plot(ylim, ylim, 'k')
ax2.plot(x[t_plot, 1], x_pc_bar[t_plot, 1], 'r.')
ax2.set_xlabel('$x_2$')
ax2.set_ylabel('$\overline{x}_{2}^{pc}$')
ax2.set_xlim(ylim)
ax1.set_ylim(ylim)
add_logo(fig2, size_frac_x=1/4)
plt.tight_layout()

fig3, axes = plt.subplots(2, 4)
for i in range(2):
    for j in range(2):
        axes[i, j].plot(omega_vec, np.real(ktilde_x[:, i, j]), 'b')
        axes[i, j].plot(omega_vec, np.imag(ktilde_x[:, i, j]), 'r')
        axes[i, j].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axes[i, j].set_xticklabels(['$-\pi$', '$-\pi/2$',
                                    '$0$', '$\pi$', '$\pi/2$'])
        axes[i, j].set_ylabel(r'$[\tilde{k}_x(\omega)]_{'+str(i+1)+str(j+1)+'}$')
    for j in range(2):
        axes[i, j+2].plot(omega_vec, np.real(ktilde_x_pc_bar[:, i, j]), 'b')
        axes[i, j+2].plot(omega_vec, np.imag(ktilde_x_pc_bar[:, i, j]), 'r')
        axes[i, j+2].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axes[i, j+2].set_xticklabels(['$-\pi$', '$-\pi/2$',
                                     '$0$', '$\pi$', '$\pi/2$'])
        axes[i, j+2].set_ylabel(r'$[\tilde{k}_{\bar{x}}(\omega)]^{pc}_{'+str(i+1)+str(j+1)+'}$')
add_logo(fig3, size_frac_x=1/4, location=1)
plt.tight_layout()
