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

# # s_cop_marg_stresstest [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cop_marg_stresstest&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=sdoc-copula-stresstest).

# +
import numpy as np
from scipy.stats import lognorm, gamma
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from arpym.statistics import simulate_t, quantile_sp, cop_marg_sep,\
                                cop_marg_comb
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_stresstest-parameters)

j_ = 10**4  # number of scenarios
n_ = 30  # dimension of the target X
mu_eps = np.zeros(n_)  # location of epsilon
sigma2_eps = np.eye(n_)  # dispersion of epsilon
nu_eps = 20  # dof of epsilon
k_ = 15  # dimension of the factors Z
mu_z = np.zeros(k_)  # location of Z
sigma2_z = np.eye(k_)  # dispersion of Z
nu_z = 5  # dof of Z
b1 = toeplitz(np.linspace(-0.9, 1.1, n_), np.linspace(-0.6, 1.2, k_))
b2 = toeplitz(np.linspace(-2, 0.5, n_), np.linspace(-0.7, 1, k_))
b = b1 + np.sin(b1@((b2.T@(b1@b2.T))@b1))
mu_1 = 0.2  # lognormal location
sigma2_1 = 0.25  # lognormal scale parameter
k_grid = np.linspace(1, 10, (n_-1))  # Gamma degree of freedom
theta_grid = np.linspace(1, 20, (n_-1))  # Gamma scale parameter

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_stresstest-implementation-step00): Generate scenarios for target variable with equal probabilities

z = simulate_t(mu_z, sigma2_z, nu_z, j_)
eps = simulate_t(mu_eps, sigma2_eps, nu_eps, j_)
y = z@b.T + eps
p = np.ones(j_)/j_  # flat flexible probabilities

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_stresstest-implementation-step01): Separation step

u, y_sort, cdf_y = cop_marg_sep(y, p=p)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_stresstest-implementation-step02): Marginal cdf's

# +
# lognormal marginal
cdf_x_l = lognorm.cdf(y_sort[:, 0], np.sqrt(sigma2_1), np.exp(mu_1))

cdf_x_g = np.zeros((j_, (n_-1)))
for n in range((n_-1)):
    # Gamma marginals
    cdf_x_g[:, n] = gamma.cdf(y_sort[:, n], k_grid[n], scale=theta_grid[n])

cdf_x = np.c_[cdf_x_l, cdf_x_g]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_stresstest-implementation-step03): Combination step

x = cop_marg_comb(u, y_sort, cdf_x)

# ## Plots

# +

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

plt.style.use('arpm')

# Colors
y_color = [153/255, 205/255, 129/255]
u_color = [60/255, 149/255, 145/255]
x_color = [4/255, 63/255, 114/255]

# Copula-marginal separation

# Figure specifications
plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
gs0 = gridspec.GridSpec(2, 2)

xlim = [np.percentile(y[:, 0], 0.5), np.percentile(y[:, 0], 99.5)]
ylim = [np.percentile(y[:, 1], 0.5), np.percentile(y[:, 1], 99.5)]
u_lim = [0, 1]

# Marginal X1
gs00 = gridspec.GridSpecFromSubplotSpec(23, 20, subplot_spec=gs0[0])
ax1 = plt.Subplot(f, gs00[:-5, 4:-5], ylim=u_lim)
f.add_subplot(ax1)
ax1.tick_params(labelsize=14)
ax1.set_xlim([-20, 20])
plt.plot(y_sort[:, 0], cdf_y[:, 0], lw=2, color=y_color)
plt.title(r'Distribution of $Y_1$', fontsize=20, fontweight='bold', y=1.03)

# Copula scenarios
gs01 = gridspec.GridSpecFromSubplotSpec(46, 18, subplot_spec=gs0[1],
                                        wspace=0, hspace=0.6)
ax2 = plt.Subplot(f, gs01[:-10, 3:-8], ylim=[0, 1], xlim=[0, 1])
f.add_subplot(ax2)
plt.scatter(u[:, 1], u[:, 0], s=5, color=u_color)
ax2.tick_params(labelsize=14)
plt.title(r'Copula $\mathbf{U}$', fontsize=20, fontweight='bold', y=1.03)
ax2_txt = ax2.text(0.1, 0.9 ,"",fontsize=20)
ax2_title_1 = r'$\mathrm{\mathbb{C}}$'+r'r'+r"$\{U_1,U_2\}=%2.2f$" % (np.corrcoef(u[:,:2].T)[0,1])
ax2_txt.set_text(ax2_title_1)

# Grade U1
ax3 = plt.Subplot(f, gs01[:-10, 1])
ax3.tick_params(labelsize=14)
f.add_subplot(ax3)
plt.xlim([0, 2])
plt.ylim([0, 1])
ax3.get_yaxis().set_visible(False)
plt.hist(np.sort(u[:, 0]), weights=p, bins=int(10*np.log(j_)), density=True,
         color=u_color, orientation='horizontal')
plt.title('Grade $U_1$', fontsize=16, fontweight='bold', y=1.03)

# Grade U2
ax4 = plt.Subplot(f, gs01[41:46, 3:-8], sharex=ax2)
f.add_subplot(ax4)
ax4.tick_params(labelsize=14)
ax4.get_xaxis().set_visible(False)
plt.hist(np.sort(u[:, 1]), weights=p, bins=int(10*np.log(j_)),
         density=True, color=u_color)
ax4.set_title('Grade $U_2$', fontsize=16, fontweight='bold', x=-0.27, y=0)
ax4.yaxis.tick_right()
plt.ylim([0, 2])
plt.xlim([0, 1])

# Joint scenarios
gs02 = gridspec.GridSpecFromSubplotSpec(24, 20, subplot_spec=gs0[2], wspace=0.2, hspace=0.5)
ax5 = plt.Subplot(f, gs02[7:, 4:-5])
f.add_subplot(ax5)
plt.scatter(y[:, 0], y[:, 1], s=5, color=y_color, label=r'$F_{X_{1}}(x)$')
ax5.set_xlim([-20, 20])
ax5.set_ylim([-8, 8])
ax5.tick_params(labelsize=14)
plt.xlabel('$Y_1$', fontsize=17)
plt.ylabel('$Y_2$', fontsize=17)
ax5_title = 'Joint'+r' $\mathbf{Y}=\mathbf{\beta}\mathbf{Z}  + \mathbf{\varepsilon}$'
plt.title(ax5_title, fontsize=20, fontweight='bold', y=-0.3)
ax5_txt = ax5.text(-7, 6.5 ,"",fontsize=20)
ax5_title_1 = r'$\mathrm{\mathbb{C}}$'+r'r'+r"$\{Y_1,Y_2\}=%2.2f$" % (np.corrcoef(y[:,:2].T)[0,1])
ax5_txt.set_text(ax5_title_1)


# Histogram Y1
ax7 = plt.Subplot(f, gs02[0:5, 4:-5])
f.add_subplot(ax7)
plt.hist(y[:, 0], weights=p, bins=int(20*np.log(j_)), density=True, color=y_color)
ax7.tick_params(labelsize=14)
ax7.set_ylim([0, 0.45])
ax7.set_xlim([-20, 20])
ax7.get_xaxis().set_visible(False)


# Histogram Y2
ax8 = plt.Subplot(f, gs02[7:, -4:-1])
f.add_subplot(ax8)
plt.hist(y[:, 1], weights=p, bins=int(20*np.log(j_)), density=True,
         orientation='horizontal', color=y_color)
ax8.tick_params(labelsize=14)
ax8.set_xlim([0, 0.4])
ax8.set_ylim([-8, 8])
ax8.get_yaxis().set_visible(False)


# Marginal Y2
gs03 = gridspec.GridSpecFromSubplotSpec(25, 18, subplot_spec=gs0[3])
ax6 = plt.Subplot(f, gs03[7:, 3:-8])
f.add_subplot(ax6)
plt.plot(cdf_y[:, 1], y_sort[:, 1], lw=2, color=y_color)
plt.title(r'Distribution of $Y_2$', fontsize=20, fontweight='bold', y=-0.3)
ax6.tick_params(labelsize=14)
ax6.set_ylim([-8, 8])
plt.xlim([0, 1])

add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()

# Copula-marginal combination

plt.style.use('arpm')

y_lim = [np.percentile(x[:, 0], 0.5), np.percentile(x[:, 0], 99.5)]
x_lim = [np.percentile(x[:, 1], 0.5), np.percentile(x[:, 1], 99.5)]
u_lim = [0, 1]

plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
gs0 = gridspec.GridSpec(2, 2)

# # Marginal X2
gs00 = gridspec.GridSpecFromSubplotSpec(44, 18, subplot_spec=gs0[0],
                                        wspace=2, hspace=2.5)
ax1 = plt.Subplot(f, gs00[:-15, 4:-6], ylim=x_lim, xlim=[0, 1])
f.add_subplot(ax1)
plt.plot(np.sort(u[:, 1]), gamma.ppf(np.sort(u[:, 1]), k_grid[0],
                 scale=theta_grid[0]), lw=2, color=x_color)
ax1.invert_xaxis()
ax1.tick_params(labelsize=14)
plt.title('Distribution of $X_2$', fontsize=20, fontweight='bold')

# Copula scenarios
gs01 = gridspec.GridSpecFromSubplotSpec(46, 18, subplot_spec=gs0[2], wspace=2)
ax2 = plt.Subplot(f, gs01[8:-3, 4:-6], ylim=[0, 1], xlim=[0, 1])
f.add_subplot(ax2)
anim3 = plt.scatter(u[:, 1], u[:, 0], s=5, color=u_color)
ax2.tick_params(labelsize=14)
ax2_txt = ax2.text(0, 0.89 ,"", fontsize=20)
ax2_title_1 = r'$\mathrm{\mathbb{C}}$'+r'r'+r"$\{U_1,U_2\}=%2.2f$" % (np.corrcoef(u[:, :2].T)[0, 1])
ax2_txt.set_text(ax2_title_1)

ax3 = plt.Subplot(f, gs01[8:-3, -6:-4], sharey=ax2)
f.add_subplot(ax3)
plt.title('Grade $U_1$', fontsize=16, fontweight='bold')
ax3.get_yaxis().set_visible(False)
plt.hist(np.sort(u[:, 0]), weights=p, bins=int(10*np.log(j_)), density=True,
                                    orientation='horizontal', color=u_color)
ax3.tick_params(labelsize=14)
plt.xlim([0, 2])
plt.ylim([0, 1])

ax4 = plt.Subplot(f, gs01[0:6, 4:-6], sharex=ax2)
f.add_subplot(ax4)
ax4.get_xaxis().set_visible(False)
plt.hist(u[:, 1], weights=p, bins=int(10*np.log(j_)), density=True, color=u_color)
plt.title('Grade $U_2$', fontsize=16, fontweight='bold')
ax4.tick_params(labelsize=14)
plt.xlim([0, 1])
plt.ylim([0, 2])

# Joint scenarios
gs02 = gridspec.GridSpecFromSubplotSpec(44, 40, subplot_spec=gs0[1],
                                        wspace=0.6, hspace=1)
ax5 = plt.Subplot(f, gs02[:-15, 8:-8])
f.add_subplot(ax5)
plt.scatter(x[:, 0], x[:, 1], s=5, color=x_color)
ax5.tick_params(labelsize=14)
plt.title(r"Joint $\mathbf{X}$ $=CopMarg(f_{\mathbf{U}}, \{f_{X_n}\}_{n=1}^{\bar{n}})$", fontsize=20, fontweight='bold', y=1.05)
ax5.set_xlim([1, 10])
ax5.set_ylim(x_lim)
plt.xlabel('$X_1$', labelpad=-5, fontsize=17)
plt.ylabel('$X_2$', fontsize=17)
ax5_txt = ax5.text(6.5, 4, "", fontsize=20)
ax5_title_1 = r'$\mathrm{\mathbb{C}}$'+r'r'+r"$\{X_1,X_2\}=%2.2f$" % (np.corrcoef(x[:,:2].T)[0,1])
ax5_txt.set_text(ax5_title_1)

# X1
ax7 = plt.Subplot(f, gs02[-11:-1, 8:-8])
f.add_subplot(ax7)
ax7.get_xaxis().set_visible(False)
ax7.invert_yaxis()
ax7.set_xlim([1, 10])
ax7.set_ylim([0, 0.05])
plt.hist(np.sort(x[:, 0]), weights=p, bins=int(120*np.log(j_)),
                                              color=x_color, bottom=0)
ax7.tick_params(labelsize=14)
plt.gca().invert_yaxis()

# X2
ax8 = plt.Subplot(f, gs02[:-15, 1:6])
f.add_subplot(ax8)
ax8.get_yaxis().set_visible(False)
plt.hist(np.sort(x[:, 1]), weights=p, bins=int(30*np.log(j_)),
         orientation='horizontal', color=x_color, bottom=0)
ax8.set_xlim([0, 0.1])
plt.gca().invert_xaxis()


# Marginal X1
gs03 = gridspec.GridSpecFromSubplotSpec(46, 40, subplot_spec=gs0[3])
ax6 = plt.Subplot(f, gs03[8:-3,  8:-8], xlim=[1, 10], ylim=[0, 1])
f.add_subplot(ax6)
ax6.set_xlim([1, 10])
ax6.tick_params(labelsize=14)
plt.plot(lognorm.ppf(np.sort(u[:, 0]), sigma2_1, np.exp(mu_1)),
         np.sort(u[:, 0]), lw=2, color=x_color)
plt.title('Distribution of $X_1$', fontsize=20, fontweight='bold')

add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()
