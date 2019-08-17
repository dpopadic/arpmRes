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

# # s_continuum_discrete_generative_pred [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_continuum_discrete_generative_pred&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_generative_pred).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_generative_pred-parameters)

p = 0.6  # unconditional probability of the true dist
q = 0.5  # unconditional probability of the model
mu_x_0 = 3.5  # conditional expectation of the true distribution
mu_x_1 = 6  # conditional expectation of the true distribution
m_0 = 3  # conditional expectation of the model
m_1 = 5.5  # conditional expectation of the model
sig2_x_0 = 1.21  # conditional variance
sig2_x_1 = 0.64  # conditional variance
j_ = 10**5  # number of simulations

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_generative_pred-implementation-step01): Generate samples

# +
z = np.random.binomial(1, p, 100)
x = (1-z)*simulate_normal(mu_x_0, sig2_x_0, 100) +\
    z*simulate_normal(mu_x_1, sig2_x_1, 100)
z_q = np.random.binomial(1, q, j_)
x_q = (1-z_q)*simulate_normal(m_0, 1, j_) + z_q*simulate_normal(m_1, 1, j_)

no_points_grid = 500
x_grid = np.linspace(min(np.percentile(x, 1), np.percentile(x_q, 1)),
                     max(np.percentile(x, 99), np.percentile(x_q, 99)),
                     no_points_grid)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_continuum_discrete_generative_pred-implementation-step02): Compute expected score of the model

# +

def norm_pdf(x, mu, sig):
    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-((x-mu)**2/(2*sig**2)))


# postulated model
def f_m0m1q(x, z):
    return q**z*(1-q)**(1-z)*norm_pdf(x-m_0*(1-z)-m_1*z, 0, 1)


exp_log_score = np.mean(-np.log(f_m0m1q(x, z)))
# -

# ## Plots

# +
plt.style.use('arpm')
# colors
teal = [0.2344, 0.582, 0.5664]
light_green_2 = [0.4781, 0.6406, 0.4031]
light_grey = [0.4, 0.4, 0.4]
colf = [0, 0.5412, 0.9020]
markersize = 60
j_plot = 10**2  # number of plotted simulations
# X|z=0 and X|z=1 pdf

fx_0 = norm_pdf(x_grid, mu_x_0, sig2_x_0**0.5)
fx_1 = norm_pdf(x_grid, mu_x_1, sig2_x_1**0.5)

fig = plt.figure(dpi=72)
# plot locations
pos1 = [0.346, 0.2589, 0.56888, 0.7111]
pos2 = [0.336, 0.03, 0.56888, 0.1889]
pos3 = [0.157, 0.2589, 0.16, 0.7111]
pos4 = [0, 0.2589, 0.08, 0.1889]
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
# lines through means
ax1.plot(xlimm, [mu_x_0, mu_x_0], xlimm, [mu_x_1, mu_x_1],
         c=light_grey, lw=0.5)
# joint
l5 = ax1.scatter(z, x, s=markersize*3,
                 edgecolor=light_grey, c='none', marker='o')
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
# Bernoulli(p), Bernoulli(q)
l3 = ax2.bar(0.1, 1-p, .2, color='none', edgecolor=teal, lw=2, align='center')
l4 = ax2.bar(0.1, 1-q, 0.1, color=light_green_2, alpha=0.7, align='center')
ax2.bar(1.1, p, 0.2, bottom=1-p, color='none',
        edgecolor=teal, lw=2, align='center')
ax2.bar(1.1, q, 0.1, bottom=1-q,
        color=light_green_2, alpha=0.7, align='center')
ax2.plot([0.15, 1.05], [1-q, 1-q], c=light_green_2, lw=0.5)
ax2.plot([0.2, 1], [1-p, 1-p], c=teal, lw=0.5)
plt.box(False)

# left plot
ax3 = fig.add_axes(pos3)
ax3.set_xlim([0, np.max([fx_0, fx_1])])
ax3.set_ylim([x_grid[0], x_grid[-1]])
ax3.set_facecolor('none')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.grid(False)
ax3.invert_xaxis()
ax3.hist(x_q, bins='auto', density=True, facecolor=light_green_2,
         orientation='horizontal')
l1, = ax3.plot((1-p)*fx_0 + p*fx_1, x_grid, color=teal, lw=2)
plt.box(False)

# Expected score plot
ax4 = fig.add_axes(pos4)
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1.05*exp_log_score])
ax4.set_facecolor('none')
ax4.grid(True, color=[0.9, 0.9, 0.9])
ax4.set_xticks([])
ax4.bar(0.5, exp_log_score, 1, color=colf, align='center')
ax4.set_title('Expected score',
              fontdict={'fontsize': '17', 'fontweight': 'bold'},
              loc='left')

# dummy plot for histogram
l2 = Rectangle((0, 0), 1, 1, color=light_green_2, ec='none')
fig.legend((l1, l2, l3, l4, l5),
           ('Marginal X', 'Marginal X model',
            'Marginal Z', 'Marginal Z model', 'Joint (X,Z)'), 'lower left',
           prop={'size': '17', 'weight': 'bold'},
           facecolor='none', edgecolor='none')

add_logo(fig)
