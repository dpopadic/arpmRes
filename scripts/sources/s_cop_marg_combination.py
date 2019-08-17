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

# # s_cop_marg_combination [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_cop_marg_combination&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-norm-cop-giv-marg).

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc, rcParams

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{amssymb}"]

from scipy.stats import lognorm, gamma
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_combination-parameters)

mu_1 = 0.2  # lognormal location
sigma2_1 = 0.25  # lognormal scale parameter
k_2 = 1  # Gamma degree of freedom
theta_2 = 1  # Gamma scale parameter

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_combination-implementation-step01): Load copula-scenarions from the separation step

# +
data = pd.read_csv('../../../databases/temporary-databases/db_separation_data.csv')

j_ = pd.read_csv('../../../databases/temporary-databases/db_separation_data.csv',
                 usecols=['j_'], nrows=1).values[0, 0].astype(int)
n_ = pd.read_csv('../../../databases/temporary-databases/db_separation_data.csv',
                 usecols=['n_'], nrows=1).values[0, 0].astype(int)
u = pd.read_csv('../../../databases/temporary-databases/db_separation_data.csv',
                usecols=['u']).values.reshape(j_, n_)
p = np.ones(j_)/j_
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_cop_marg_combination-implementation-step02): Combination step

x = np.zeros((j_, 2))
x[:, 0] = lognorm.ppf(u[:, 0], np.sqrt(sigma2_1), np.exp(mu_1))
x[:, 1] = gamma.ppf(u[:, 1], k_2, scale=theta_2)

# ## Plots

# +
plt.style.use('arpm')

# Colors
y_color = [153/255, 205/255, 129/255]
u_color = [60/255, 149/255, 145/255]
x_color = [4/255, 63/255, 114/255]
m_color = [63/255, 0/255, 102/255]

# Copula-marginal combination

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
plt.plot(np.sort(u[:, 1]), gamma.ppf(np.sort(u[:, 1]), k_2, scale=theta_2),
         lw=2, color=x_color)
ax1.invert_xaxis()
ax1.tick_params(labelsize=14)
plt.ylabel('$q_{X_2}$', fontsize=17)

# Copula scenarios
gs01 = gridspec.GridSpecFromSubplotSpec(46, 18, subplot_spec=gs0[2], wspace=2)
ax2 = plt.Subplot(f, gs01[8:-3, 4:-6], ylim=[0, 1], xlim=[0, 1])
f.add_subplot(ax2)
anim3 = plt.scatter(u[:, 1], u[:, 0], s=5, color=u_color)
ax2.tick_params(labelsize=14)
ax2_txt = ax2.text(0, 0.89, "", fontsize=20, color=m_color)
ax2_title_1 = r'$\mathbb{C}$'+ r'$r$' + r"$\{U_1,U_2\}=%2.2f$" % (np.corrcoef(u[:, :2].T)[0, 1])
ax2_txt.set_text(ax2_title_1)
plt.xlabel('$U_2$', fontsize=17, labelpad=-10)
plt.ylabel('$U_1$', fontsize=17, labelpad=-10)
ax2_title = r'Copula ' + r'$\boldsymbol{U}$'
ax2.set_title(ax2_title, fontsize=20, y=-0.2, fontweight='bold')

ax3 = plt.Subplot(f, gs01[8:-3, -6:-4], sharey=ax2)
f.add_subplot(ax3)
plt.hist(np.sort(u[:, 0]), weights=p, bins=int(10*np.log(j_)), density=True,
         orientation='horizontal', color=u_color)
ax3.tick_params(labelsize=14)
ax3.tick_params(axis='y', colors='None')
plt.xlim([0, 2])
plt.ylim([0, 1])
plt.xlabel('$f_{U_1}$', fontsize=17, labelpad=-10)

ax4 = plt.Subplot(f, gs01[0:6, 4:-6], sharex=ax2)
f.add_subplot(ax4)
plt.hist(u[:, 1], weights=p, bins=int(10*np.log(j_)),
         density=True, color=u_color)
ax4.tick_params(labelsize=14)
ax4.tick_params(axis='x', colors='None')
plt.xlim([0, 1])
plt.ylim([0, 2])
plt.ylabel('$f_{U_2}$', fontsize=17)

# Joint scenarios
gs02 = gridspec.GridSpecFromSubplotSpec(44, 40, subplot_spec=gs0[1],
                                        wspace=0.6, hspace=1)
ax5 = plt.Subplot(f, gs02[:-15, 8:-8])
f.add_subplot(ax5)
plt.scatter(x[:, 0], x[:, 1], s=5, color=x_color)
ax5.tick_params(labelsize=14)
ax5.set_xlim(y_lim)
ax5.set_ylim(x_lim)
plt.xlabel('$X_1$', labelpad=-5, fontsize=17)
plt.ylabel('$X_2$', fontsize=17)
ax5_txt = ax5.text(3.2, 4.5, "", fontsize=20, color=m_color)
ax5_title_1 = r'$\mathbb{C}$'+ r'$r$' + r"$\{X_1,X_2\}=%2.2f$" % (np.corrcoef(x[:, :2].T)[0, 1])
ax5_txt.set_text(ax5_title_1)
plt.title(r"Joint $\boldsymbol{X}$", fontsize=20, fontweight='bold', y=1.05)

# X1
ax7 = plt.Subplot(f, gs02[-11:-1, 8:-8])
f.add_subplot(ax7)
ax7.tick_params(axis='x', colors='None')
ax7.set_xlim(y_lim)
ax7.set_ylim([0, 0.05])
plt.hist(np.sort(x[:, 0]), weights=p, bins=int(120*np.log(j_)),
         color=x_color, bottom=0)
ax7.tick_params(labelsize=14)
plt.gca().invert_yaxis()
plt.ylabel('$f_{X_1}$', fontsize=17)

# X2
ax8 = plt.Subplot(f, gs02[:-15, 1:6])
f.add_subplot(ax8)
plt.hist(np.sort(x[:, 1]), weights=p, bins=int(30*np.log(j_)),
         orientation='horizontal', color=x_color, bottom=0)
ax8.set_xlim([0, 0.1])
ax8.set_ylim(x_lim)
ax8.tick_params(axis='y', colors='None')
plt.gca().invert_xaxis()
plt.xlabel('$f_{X_2}$', fontsize=17)
ax8.xaxis.tick_top()

# Marginal X1
gs03 = gridspec.GridSpecFromSubplotSpec(46, 40, subplot_spec=gs0[3])
ax6 = plt.Subplot(f, gs03[8:-3,  8:-8], xlim=[1, 10], ylim=[0, 1])
f.add_subplot(ax6)
plt.plot(lognorm.ppf(np.sort(u[:, 0]), np.sqrt(sigma2_1), np.exp(mu_1)),
         np.sort(u[:, 0]), lw=2, color=x_color)
ax6.set_xlim(y_lim)
ax6.tick_params(labelsize=14)
plt.xlabel('$q_{X_1}$', fontsize=17, labelpad=-5)

add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()
