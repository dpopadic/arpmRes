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

# # s_cop_marg_separation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cop_marg_separation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_separation).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc, rcParams

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{amssymb}"]

from arpym.statistics import cop_marg_sep, simulate_t
import scipy as sp
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_separation-parameters)

j_ = 10**5  # number of scenarios
mu_eps = np.zeros(2)  # location of epsilon
sigma2_eps = np.eye(2)  # dispersion of epsilon
nu_eps = 5  # dof of epsilon
mu_z = np.zeros(1)  # location of Z
sigma2_z = np.eye(1)  # dispersion of Z
nu_z = 2  # dof of Z
b = np.array([[np.cos(1.8)], [np.sin(0.1)]])  # loadings

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_separation-implementation-step00): Generate scenarios for target variable with equal probabilities

z = simulate_t(mu_z, sigma2_z, nu_z, j_).reshape((j_, -1))
eps = simulate_t(mu_eps, sigma2_eps, nu_eps, j_).reshape((j_, -1))
x = z@b.T + eps
p = np.ones(j_)/j_

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_separation-implementation-step01): Separation step

u, x_sort, cdf_x = cop_marg_sep(x, p=p)
cdf_x_tilde1 = sp.interpolate.interp1d(x_sort[:, 0], cdf_x[:, 0],
                                       kind='linear')
cdf_x_tilde2 = sp.interpolate.interp1d(x_sort[:, 1], cdf_x[:, 1],
                                       kind='linear')

# ## Save the data

# +
output = {
          'j_': pd.Series(j_),
          'n_': pd.Series(u.shape[1]),
          'u': pd.Series(u.reshape(-1))
          }

df = pd.DataFrame(output)
df.to_csv('../../../databases/temporary-databases/db_separation_data.csv')
# -

# ## Plot

# +
plt.style.use('arpm')

# Colors
y_color = [153/255, 205/255, 129/255]
u_color = [60/255, 149/255, 145/255]
x_color = [4/255, 63/255, 114/255]
m_color = [63/255, 0/255, 102/255]

# Figure specifications
plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
gs0 = gridspec.GridSpec(2, 2)

xlim = [np.percentile(x[:, 0], 0.5), np.percentile(x[:, 0], 99.5)]
ylim = [np.percentile(x[:, 1], 0.5), np.percentile(x[:, 1], 99.5)]
u_lim = [0, 1]
y1_plot = np.linspace(x_sort[0, 0], x_sort[-1, 0], 10**5)
y2_plot = np.linspace(x_sort[0, -1], x_sort[-1, 1], 10**5)

# Marginal X1
gs00 = gridspec.GridSpecFromSubplotSpec(23, 20,
                  subplot_spec=gs0[0], wspace=2, hspace=2.5)
ax1 = plt.Subplot(f, gs00[:-5, 4:-4], ylim=u_lim, xlim=xlim)
f.add_subplot(ax1)
ax1.tick_params(labelsize=14)
plt.plot(y1_plot, cdf_x_tilde1(y1_plot), lw=2, color=y_color)
plt.ylabel('$F_{X_1}$', fontsize=17)

# Copula scenarios
gs01 = gridspec.GridSpecFromSubplotSpec(46, 18, subplot_spec=gs0[1],
                                        wspace=0, hspace=0.6)
ax2 = plt.Subplot(f, gs01[:-10, 4:-5], ylim=[0, 1], xlim=[0, 1])
f.add_subplot(ax2)
plt.scatter(u[:, 1], u[:, 0], s=5, color=u_color)
ax2.tick_params(labelsize=14)
plt.title(r'Copula $\boldsymbol{U}$', fontsize=20, fontweight='bold', y=1.03)
ax2_txt = ax2.text(0.1, 0.9, "", fontsize=20, color=m_color)
ax2_title_1 = r'$\mathbb{C}$' + r'$r$' + r"$\{U_1,U_2\}=%2.2f$" % (np.corrcoef(u[:, :2].T)[0, 1])
ax2_txt.set_text(ax2_title_1)
plt.xlabel('$U_2$', fontsize=17, labelpad=-8)
plt.ylabel('$U_1$', fontsize=17, labelpad=-10)

# Grade U1
ax3 = plt.Subplot(f, gs01[:-10, 2])
f.add_subplot(ax3)
ax3.tick_params(labelsize=14)
plt.xlim([0, 2])
plt.ylim([0, 1])
ax3.tick_params(axis='y', colors='None')
plt.hist(np.sort(u[:, 0]), weights=p, bins=int(10*np.log(j_)), density=True,
         color=u_color, orientation='horizontal')
plt.xlabel('$f_{U_1}$', fontsize=17)
ax3.xaxis.tick_top()

# Grade U2
ax4 = plt.Subplot(f, gs01[41:46, 4:-5], sharex=ax2)
f.add_subplot(ax4)
plt.hist(np.sort(u[:, 1]), weights=p, bins=int(10*np.log(j_)),
         density=True, color=u_color)
ax4.tick_params(labelsize=14)
ax4.tick_params(axis='x', colors='white')
ax4.yaxis.tick_right()
plt.ylabel('$f_{U_2}$', fontsize=17)
plt.ylim([0, 2])
plt.xlim([0, 1])

# Joint scenarios
gs02 = gridspec.GridSpecFromSubplotSpec(2*25, 2*20,
            subplot_spec=gs0[2], wspace=0.6, hspace=1)
ax5 = plt.Subplot(f, gs02[2*7:, 2*4:-8], ylim=ylim, xlim=xlim)
f.add_subplot(ax5)
plt.scatter(x[:, 0], x[:, 1], s=5, color=y_color, label=r'$F_{X_{1}}(x)$')
ax5.tick_params(labelsize=14)
plt.xlabel('$X_1$', fontsize=17)
plt.ylabel('$X_2$', fontsize=17)
ax5_title = 'Joint' + r' $\boldsymbol{X}=\boldsymbol{\beta}Z + \boldsymbol{\varepsilon}$'
plt.title(ax5_title, fontsize=20, fontweight='bold', y=-0.3)
ax5_txt = ax5.text(-3.5, 2, "", fontsize=20, color=m_color)
ax5_title_1 = r'$\mathbb{C}$' + r'$r$' + r"$\{X_1,X_2\}=%2.2f$" % (np.corrcoef(x[:, :2].T)[0, 1])
ax5_txt.set_text(ax5_title_1)

# Histogram X1
ax7 = plt.Subplot(f, gs02[0:12, 2*4:-8], sharex=ax5)
f.add_subplot(ax7)
plt.hist(x[:, 0], weights=p, bins=int(80*np.log(j_)),
         density=True, color=y_color)
ax7.tick_params(labelsize=14)
ax7.set_ylim([0, 0.45])
ax7.set_xlim(xlim)
ax7.tick_params(axis='x', colors='None')
plt.ylabel('$f_{X_1}$', fontsize=17)

# Histogram X2
ax8 = plt.Subplot(f, gs02[2*7:, -7:-2], sharey=ax5)
f.add_subplot(ax8)
plt.hist(x[:, 1], weights=p, bins=int(80*np.log(j_)), density=True,
         orientation='horizontal', color=y_color)
ax8.tick_params(labelsize=14)
ax8.set_xlim([0, 0.4])
ax8.set_ylim(ylim)
ax8.tick_params(axis='y', colors='None')
plt.xlabel('$f_{X_2}$', fontsize=17)

# Marginal X2
gs03 = gridspec.GridSpecFromSubplotSpec(25, 18, subplot_spec=gs0[3])
ax6 = plt.Subplot(f, gs03[7:, 4:-5], sharey=ax5)
f.add_subplot(ax6)
plt.plot(cdf_x_tilde2(y2_plot), y2_plot, lw=2, color=y_color)
plt.xlabel('$F_{X_2}$', fontsize=17)
ax6.tick_params(labelsize=14)
ax6.set_ylim(ylim)
plt.xlim([0, 1])

add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()
