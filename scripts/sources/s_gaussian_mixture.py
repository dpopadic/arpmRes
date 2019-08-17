#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_gaussian_mixture [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_gaussian_mixture&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_gaussian_mixture).

# +
import numpy as np
from scipy.special import logit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_gaussian_mixture-parameters)

p = 0.6  # unconditional probability
mu_x_0 = 3.5  # conditional expectation
mu_x_1 = 6  # conditional expectation
sig2_x_0 = 1.21  # conditional variance
sig2_x_1 = 0.64  # conditional variance
x_cond = 5.5  # realization of X
j_ = 10**5  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_gaussian_mixture-implementation-step01): Generate samples

z = np.random.binomial(1, p, j_)
x = (1-z)*simulate_normal(mu_x_0, sig2_x_0, j_) +\
    z*simulate_normal(mu_x_1, sig2_x_1, j_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_gaussian_mixture-implementation-step02): Compute Z|x

# +
alpha = logit(p) - 0.5*(np.log(sig2_x_1)-np.log(sig2_x_0) +
              mu_x_1/sig2_x_1*mu_x_1 - mu_x_0/sig2_x_0*mu_x_0)
beta = mu_x_1/sig2_x_1 - mu_x_0/sig2_x_0
gamma = -0.5*(1/sig2_x_1 - 1/sig2_x_0)

def logit_px(x): return  alpha + beta*x + gamma*x**2
def p_x_func(x): return 1 / (1 + np.exp(-logit_px(x)))
p_x_cond = p_x_func(x_cond)
# -

# ## Plots

# +
plt.style.use('arpm')
# colors
teal = [0.2344, 0.582, 0.5664]
light_green_2 = [0.4781, 0.6406, 0.4031]
light_grey = [0.4, 0.4, 0.4]
markersize = 60
j_plot = 10**2  # number of plotted simulations

no_points_grid = 500
x_grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), no_points_grid)

def norm_pdf(x, mu, sig):
    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-((x-mu)**2/(2*sig**2)))
fx_0 = norm_pdf(x_grid, mu_x_0, sig2_x_0**0.5)
fx_1 = norm_pdf(x_grid, mu_x_1, sig2_x_1**0.5)

p_x = p_x_func(x_grid)

fig = plt.figure(dpi=72)
fig.set_size_inches(10, 8)
# plot locations
pos1 = [0.346, 0.2589, 0.56888, 0.7111]
pos2 = [0.336, 0.03, 0.56888, 0.1889]
pos3 = [0.157, 0.2589, 0.16, 0.7111]
# top right plot
ax1 = fig.add_axes(pos1)
ax1.set_xlim([-0.07, 1.2])
ax1.set_ylim([x_grid[0], x_grid[-1]])
xlimm = ax1.get_xlim()
ylimm = ax1.get_ylim()
ax1.set_facecolor('none')
ax1.set_xticks([0, 1])
ax1.set_yticks(np.arange(np.ceil(ylimm[0]), np.floor(ylimm[1])+1))
ax1.set_xlabel('$Z$', labelpad=-30, fontsize=14)
ax1.set_ylabel('$X$', labelpad=-30, fontsize=14)
# axes
ax1.plot([0, 0], ylimm, 'k', lw=0.5)
ax1.plot(xlimm,
         [ylimm[0]+.05*(ylimm[1]-ylimm[0]), ylimm[0]+.05*(ylimm[1]-ylimm[0])],
         'k', lw=0.5)
# P{Z=1|x} on main plot
ax1.barh(x_cond, p_x_cond, (ylimm[1]-ylimm[0])*0.03, color='none',
         edgecolor=teal, lw=2, align='center')
ax1.barh(x_cond, 1, (ylimm[1]-ylimm[0])*0.03, color='none', edgecolor=teal,
         lw=1, align='center')
# lines through means
ax1.plot(xlimm, [mu_x_0, mu_x_0], xlimm, [mu_x_1, mu_x_1],
         c=light_grey, lw=0.5)
# joint
l1 = ax1.scatter(z[:j_plot], x[:j_plot], s=markersize*3,
                 edgecolor=light_grey, c='none', marker='o')
# E{X|z}
l9 = ax1.scatter([0, 1], [mu_x_0, mu_x_1], marker='x', s=markersize*3,
                 c=[light_green_2], lw=6)
# P{X=1}
l5, = ax1.plot(p_x, x_grid, ls='--', lw=2, color=teal)
# realizations of X and Z
l3 = ax1.scatter(-0.04, x_cond, marker='o', s=markersize*3, c=[teal])
l7 = ax1.scatter(1, ylimm[0]+0.02*(ylimm[1]-ylimm[0]),
                 marker='o', s=markersize*3, c=[light_green_2])
ax1.grid(False)
plt.box(False)

# bottom plot
ax2 = fig.add_axes(pos2)
ax2.set_xlim([0, 1.27])
ax2.set_ylim([-0.01, 1.03])
ax2.set_facecolor('none')
ax2.set_yticks([0, 0.5, 1])
ax2.yaxis.tick_right()
ax2.grid(True, color=[0.4, 0.4, 0.4])
ax2.set_xticks([])
# Bernoulli(p), Bernoulli(p(x_cond))
l6 = ax2.bar(0.1, 1-p, 0.2, color=light_green_2, align='center')
ax2.bar(0.1, 1-p_x_cond, 0.1, bottom=p_x_cond, color='none',
        edgecolor=teal, lw=1, align='center')
ax2.bar(1.1, p, 0.2, bottom=1-p, color=light_green_2, align='center')
l4 = ax2.bar(1.1, p_x_cond, 0.1, color='none',
             edgecolor=teal, lw=2, align='center')
ax2.plot([0.15, 1.05], [p_x_cond, p_x_cond], c=teal, lw=0.5)
ax2.plot([0.2, 1], [1-p, 1-p], c=light_green_2, lw=0.5)
plt.box(False)

# left plot
ax3 = fig.add_axes(pos3)
ax3.set_xlim([0, 1.1*np.max([fx_0, fx_1])])
ax3.set_ylim([x_grid[0], x_grid[-1]])
ax3.set_facecolor('none')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.grid(False)
ax3.invert_xaxis()
# pdf's X|z
ax3.plot(fx_0, x_grid, c=light_green_2, lw=2)
l8, = ax3.plot(fx_1, x_grid, c=light_green_2, lw=2)
# marginal X
ax3.hist(x, bins='auto', density=True, facecolor=teal,
         orientation='horizontal')
plt.box(False)

l2 = Rectangle((0, 0), 1, 1, color=teal, ec='none')  # dummy plot for histogram
fig.legend((l1, l2, l3, l4, l5, l6, l7, l8, l9),
           ('Joint (X,Z)', 'Marginal X', 'Realization x', 'Conditional Z|x',
            'Conditional P{Z=1|x}', 'Marginal Z', 'Realization z',
            'Conditional X|z', 'Conditional E{X|z}'), 'lower left',
           prop={'size': '17', 'weight': 'bold'},
           facecolor='none', edgecolor='none')

add_logo(fig)
