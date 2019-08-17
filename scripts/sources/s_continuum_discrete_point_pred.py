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

# # s_continuum_discrete_point_pred [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_continuum_discrete_point_pred&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_point_pred).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_point_pred-parameters)

p = 0.6  # unconditional probability
mu_x_0 = 0.45  # conditional expectation
mu_x_1 = 0.75  # conditional expectation
sig2_x_0 = 0.0225  # conditional variance
sig2_x_1 = 0.01  # conditional variance
m_0 = 0.3  # arbitrary linear predictor prediction for z=0
m_1 = 1  # arbitrary linear predictor prediction for z=1
j_ = 10**5  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_point_pred-implementation-step01): Generate samples

# +
z = np.random.binomial(1, p, j_)
x = (1-z)*np.random.normal(mu_x_0, sig2_x_0**0.5, j_) +\
    z*np.random.normal(mu_x_1, sig2_x_1**0.5, j_)

no_points_grid = 500
x_grid = np.linspace(-0.07, 1.2, no_points_grid)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_point_pred-implementation-step02): Compute arbitrary and optimal predictor

# +

def chi_m0m1(z):
    x = np.nan*np.ones_like(z)
    x[z == 0] = m_0
    x[z == 1] = m_1
    return x


def chi_mu0mu1(z):
    x = np.nan*np.ones_like(z)
    x[z == 0] = mu_x_0
    x[z == 1] = mu_x_1
    return x


x_hat_arb = chi_m0m1(z)
x_hat_opt = chi_mu0mu1(z)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_point_pred-implementation-step03): Compute X|z pdf

# +

def norm_pdf(x, mu, sig):
    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-((x-mu)**2/(2*sig**2)))


fx_0 = norm_pdf(x_grid, mu_x_0, sig2_x_0**0.5)
fx_1 = norm_pdf(x_grid, mu_x_1, sig2_x_1**0.5)
# -

# ## Plots

# +
plt.style.use('arpm')

# arbitrary realizations of X for z=0,1 closest to the selected points
x_arb_0 = 0.4
x_arb_1 = 0.8

# colors
teal = [0.2344, 0.582, 0.5664]
light_green_2 = [0.4781, 0.6406, 0.4031]
light_grey = [0.4, 0.4, 0.4]
orange = [0.94, 0.35, 0]
markersize = 60
j_plot = 10**2  # number of plotted simulations

fig = plt.figure(dpi=72)
fig.set_size_inches(10, 8)
# plot locations
pos1 = [0.346, 0.2589, 0.56888, 0.7111]
pos2 = [0.336, 0.03, 0.56888, 0.1889]
pos3 = [0.157, 0.2589, 0.16, 0.7111]
# top right plot
ax1 = fig.add_axes(pos1)
ax1.axis('equal')
ax1.set_xlim([-0.07, 1.2])
ax1.set_ylim([-0.07, 1.2])
xlimm = ax1.get_xlim()
ylimm = ax1.get_ylim()
ax1.set_facecolor('none')
ax1.set_xticks([0, 1])
ax1.set_yticks(np.arange(np.ceil(ylimm[0]), np.floor(ylimm[1])+1), 0.2)
ax1.set_xlabel('$Z$', labelpad=-30, fontsize=14)
ax1.set_ylabel('$X$', labelpad=-30, fontsize=14)
# axes
ax1.plot([0, 0], ylimm, 'k', lw=0.5)
ax1.plot(xlimm,
         [ylimm[0]+.05*(ylimm[1]-ylimm[0]), ylimm[0]+.05*(ylimm[1]-ylimm[0])],
         'k', lw=0.5)
# lines through means
ax1.plot(xlimm, [mu_x_0, mu_x_0], xlimm, [mu_x_1, mu_x_1],
         c=light_grey, lw=0.5)
# joint
l1 = ax1.scatter(z[:j_plot], x[:j_plot], s=markersize*3,
                 edgecolor=light_grey, c='none', marker='o')
# E{X|z}
l4 = ax1.scatter([0, 1], [mu_x_0, mu_x_1], marker='x', s=markersize*3,
                 c=orange, lw=6)
# realization of Z
l7 = ax1.scatter(1, ylimm[0]+0.02*(ylimm[1]-ylimm[0]),
                 marker='o', s=markersize*3, c=light_green_2)
# arbitrary predictor
l9 = ax1.scatter([0, 1], [m_0, m_1], marker='o', s=markersize*3,
                 c='b', lw=6)
# arbitrary simulations closest to x_arb_0 and x_arb_1
index0 = np.argmin(abs(x[z == 0]-x_arb_0))
x_0_arb = x[z == 0][index0]
index1 = np.argmin(abs(x[z == 1]-x_arb_1))
x_1_arb = x[z == 1][index1]
ax1.scatter([0, 1], [x_0_arb, x_1_arb], marker='o', s=markersize*3,
            c='k', lw=6)
# blue and red rectangles
min1 = min(x_0_arb, m_0)
br0 = Rectangle((0, min1), abs(x_0_arb-m_0), abs(x_0_arb-m_0), fill=True,
                alpha=0.2, facecolor='b', edgecolor='b')
min1 = min(x_1_arb, m_1)
br1 = Rectangle((1, min1), -abs(x_1_arb-m_1), abs(x_1_arb-m_1), fill=True,
                alpha=0.2, facecolor='b', edgecolor='b')
min1 = min(x_0_arb, mu_x_0)
rr0 = Rectangle((0, min1), abs(x_0_arb-mu_x_0), abs(x_0_arb-mu_x_0),
                fill=True, alpha=0.2, facecolor=orange, edgecolor=orange)
min1 = min(x_1_arb, mu_x_1)
rr1 = Rectangle((1, min1), -abs(x_1_arb-mu_x_1), abs(x_1_arb-mu_x_1),
                fill=True, alpha=0.2, facecolor=orange, edgecolor=orange)
ax1.add_patch(br0)
ax1.add_patch(br1)
ax1.add_patch(rr0)
ax1.add_patch(rr1)
ax1.grid(False)
plt.box(False)

# bottom plot
ax2 = fig.add_axes(pos2)
ax2.set_xlim([0, 1.27])
ax2.set_ylim([-0.01, 1.03])
ax2.set_facecolor('none')
ax2.set_yticks([0, 0.5, 1])
ax2.yaxis.tick_right()
ax2.grid(True, color=light_grey)
ax2.set_xticks([])
l2 = ax2.bar(0.1, 1-p, 0.2, color=light_green_2, align='center')
ax2.bar(1.1, p, 0.2, bottom=1-p, color=light_green_2, align='center')
ax2.plot([0.2, 1], [1-p, 1-p], c=light_green_2, lw=0.5)
plt.box(False)

# left plot
ax3 = fig.add_axes(pos3)
ax3.set_xlim([0, 1.1*np.max([fx_0, fx_1])])
ax3.set_ylim([-0.07, 1.2])
ax3.set_facecolor('none')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.grid(False)
ax3.invert_xaxis()
# pdf's X|z
ax3.plot(fx_0, x_grid, c=light_green_2, lw=2)
l3, = ax3.plot(fx_1, x_grid, c=light_green_2, lw=2)
plt.box(False)

# legend
fig.legend((l3, l4, l9, l2, l1),
           ('Conditional X|z', 'Optimal prediction', 'Arbitrary prediction',
            'Marginal Z', 'Joint (X,Z)'), 'lower left',
           prop={'size': '17', 'weight': 'bold'},
           facecolor='none', edgecolor='none')

add_logo(fig)
