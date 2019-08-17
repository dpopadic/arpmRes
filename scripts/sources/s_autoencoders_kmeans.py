#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# # s_autoencoders_kmeans [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_autoencoders_kmeans&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_autoencoders_kmeans).

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import norm
from sklearn.cluster import k_means

from arpym.statistics import meancov_sp
from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_autoencoders_kmeans-parameters)

# +
j_ = 10**5 # number of simulations
p_h = 0.4 # parameter of Bernoulli distribution
mu_x_0 = -1 # conditional expectation
mu_x_1 = 1.5  # conditional expectation
sigma2_x_0 = 0.36 # conditional variance
sigma2_x_1 = 0.49 # conditional variance
x_c = 1.5 # generic boundary point
x0 = -0.73 # generic decoder at 0
x1 = 1.92 # generic decoder at 1
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_autoencoders_kmeans-implementation-step01): Generate simulations of target variable

# +
# simulations of hidden variable
h = np.random.binomial(1, p_h, j_).reshape(-1)
# simulations of conditional target variable
x_h0 = simulate_normal(mu_x_0, sigma2_x_0, j_)
x_h1 = simulate_normal(mu_x_1, sigma2_x_1, j_)

# simulations of target variable
x = (1-h)*x_h0 + h*x_h1
# mean and variance of target variable
e_x, cv_x = meancov_sp(x)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_autoencoders_kmeans-implementation-step02): Generic encoder, generic and best decoder, errors

# +
# encoding induced by generic cluster
def zeta_c2(x):
    return x > x_c
# simulations of the generic code
z = zeta_c2(x)
# generic code probability parametar
l_=np.size(z[z==1])
p=l_/j_

# simulations of conditional variables
x_z0 = x[z==0]
x_z1 = x[z==1]
# conditional expectation and variance
e_x_z0, cv_x_z0 = meancov_sp(x_z0)
e_x_z1, cv_x_z1 = meancov_sp(x_z1)

# error of generic autoencoder
e_x_x0_z0 = meancov_sp((x_z0-x0)**2)[0]
e_x_x1_z1 = meancov_sp((x_z1-x1)**2)[0]
e_x_c_x0_x1 = (1-p)*e_x_x0_z0 + p*e_x_x1_z1

# best decoder
def chi (z):
    if z==0:
        return e_x_z0
    else:
        return e_x_z1

# within-cluster variance
e_cv_x_z = (1-p)*cv_x_z0 + p*cv_x_z1
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_autoencoders_kmeans-implementation-step03): Optimal autoencoder and error

# +
# run k_means algorithm
centroid, label, _ = k_means(x.reshape(-1, 1), 2)
# ensuring that samples are properly labeled
if (centroid[1][0] < centroid[0][0]):
    label = 1 - label
    tmp = centroid[1][0]
    centroid[1][0] = centroid[0][0]
    centroid[0][0] = tmp
# optimal boundary point of clustering
x_c_star = np.mean([centroid[0][0], centroid[1][0]])

# optimal encoder
def zeta_c2means(x):
    return x > x_c_star
# simulations of the optimal code
z_star = zeta_c2means(x)
# optimal code probability parametar
l_=np.sum(z_star==1)
p_star=l_/j_

# simulations of conditional variables
x_zstar0 = x[z_star==0]
x_zstar1 = x[z_star==1]
# conditional expectation and variance
e_x_zstar0, cv_x_zstar0 = meancov_sp(x_zstar0)
e_x_zstar1, cv_x_zstar1 = meancov_sp(x_zstar1)

# best decoder
def chi_c2means(z):
    if z==0:
        return e_x_zstar0
    else:
        return e_x_zstar1

# within-cluster variance
e_cv_x_zstar = (1-p_star)*cv_x_zstar0 + p_star*cv_x_zstar1
# -

# ## Plots

# +
plt.style.use('arpm')
# colors
teal = [0.2344, 0.582, 0.5664]
light_blue = [181/256, 225/256, 223/256]
sand = [247/256, 210/256, 169/256]
light_green_1 = [0.8398, 0.9141, 0.8125]
light_green_2 = [0.4781, 0.6406, 0.4031]
light_grey = [0.6, 0.6, 0.6]
dark_blue = [4/256, 63/256, 114/256]
dark_grey = [100/256, 100/256, 100/256]
orange = np.array([0.94, 0.35, 0])
colf = [0, 0.5412, 0.9020]
color1 = colf
color0 = dark_grey
colorc = 'aqua'
color2 = 'darkorange'
color3 = teal
#
markersize = 9
markerwidth = 4
# ensures that the resolution of the figure is 1280x720
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='white')
# plots sizes
coef1 = 280/1280
vshift = 0
initx = 0.0225
inity = 0.04
height_bottom = 0.25
height_top = (1-height_bottom-2*inity)/3-inity
height_middle = (1-height_bottom-2*inity)*2/3
width_middle = 0.57
width_left = (1-(initx+width_middle+2*initx))/4.5
hshift = width_left+initx/2
hh = 0.01
pos1 = [initx+hshift+hh, inity+height_bottom+vshift, width_middle, height_middle]
pos2 = [initx+hshift+hh, inity, width_middle, height_bottom-inity]
pos3 = [initx/2, 1-inity-height_top-height_middle-inity, width_left,
        height_middle/2*1.5]
pos4 = [initx+hshift+hh, inity+height_bottom+height_middle+inity+vshift,
        width_middle, height_top]
coef2 = (width_middle*1280)/(height_middle*720)
# number of ploted generic conditional simulations
j_plot = 100
# decide position of the zero and 1 on y axis
fake_0 = 0
fake_1 = 1
# shift for the marginal distribution
fake_shift = fake_1 + (fake_1-fake_0)*1.5
# grid for boundary points
x_c_grid_lim = [-3, 3.5]
x_c_grid_size = 500
x_c_grid = np.linspace(x_c_grid_lim[0], x_c_grid_lim[1], x_c_grid_size)

### axes 1 - Generic encoder, simulation of generic code, generic decoder, optimal deoder, illustrations of autoencoders errors

ax1 = fig.add_axes(pos1)
ax1.set_xlim(x_c_grid_lim)
ylimstart = -.5
ylim = [ylimstart, ylimstart+(x_c_grid_lim[1]-x_c_grid_lim[0])/coef2]
ax1.set_ylim(ylim)
multiplier = (fake_1-fake_0)
multipp = 0.96
scatter_shift_bigger = 0.17
scatter_shift = scatter_shift_bigger/(scatter_shift_bigger/0.06)
variance_lw = 4
zeta_lw = 2
ax1.tick_params(axis='y', right=False, labelright=False,
                labelleft=True, left=True)
plt.yticks([fake_0, fake_1], [0, 1])
ax1.set_xlabel('$x$', fontsize=20)#, labelpad=-15)
ax1.xaxis.set_label_coords(1, -0.0055)
ax1.set_ylabel('$z$', fontsize=20, rotation='horizontal')
ax1.yaxis.set_label_coords(-0.015, 1.02)
ax1.yaxis.set_label_position('right')
ax1.grid(False)
## generic encoder
hor_zeta0 = ax1.plot([x_c_grid_lim[0], x_c], [fake_0, fake_0],
                     lw=zeta_lw, color=colorc)[0]
hor_zeta1 = ax1.plot([x_c, x_c_grid_lim[1]], [fake_1, fake_1],
                     lw=zeta_lw, color=colorc)[0]
transp = 1
## best decoder for generic encoder
plot_e_x_z0 = ax1.plot(e_x_z0, fake_0-scatter_shift, 'x',
                     markeredgewidth=markerwidth, color=color2,
                     markersize=markersize)[0]
plot_e_x_z1 = ax1.plot(e_x_z1, fake_1-scatter_shift, 'x',
                     markeredgewidth=markerwidth, color=color2,
                     markersize=markersize)[0]
## arbitrary decoders
plot_x0 = ax1.plot(x0, fake_0-scatter_shift, 'x',
                  markeredgewidth=markerwidth,
                  color=color1, markersize=markersize)[0]
plot_x1 = ax1.plot(x1, fake_1-scatter_shift, 'x',
                  markeredgewidth=markerwidth,
                  color=color1, markersize=markersize)[0]
## squares
# arbitrary points in clusters for illustaration
arb_point_0 = x_c - 0.5
arb_point_1 = x_c + 1.7
arb = ax1.plot([arb_point_0], [fake_0-multipp*scatter_shift], 'o',
               color=color0, markersize=markersize-3)[0]
arbb = ax1.plot([arb_point_1], [fake_1-multipp*scatter_shift], 'o',
                color=color0, markersize=markersize-3)[0]
## squares parameters
optim_width_0 = arb_point_0 - e_x_z0
optim_width_1 = abs(e_x_z1 - arb_point_1)
arb_width_0 = arb_point_0 - x0
arb_width_1 = abs(x1 - arb_point_1)
#
square_x0, squarex_x1 = ax1.bar([x0, arb_point_1],
                           [arb_width_0, arb_width_1],
                           width=[arb_width_0, -arb_width_1],
                           bottom=[fake_0, fake_1], facecolor=color1,
                           edgecolor=color1, alpha=transp, align='edge')
square_e_x_z0, square_e_x_z1 = ax1.bar([e_x_z0, arb_point_1],
                           [optim_width_0, optim_width_1],
                           width=[optim_width_0, -optim_width_1],
                           bottom=[fake_0, fake_1], facecolor=color2,
                           edgecolor=color2, alpha=transp, align='edge')
## between-cluster standard deviation arrow
between_cl_sd= ax1.plot([], [], color=sand, lw=variance_lw)[0]
arrow_properties = dict(arrowstyle="<->", color=sand,
                        lw=variance_lw, shrinkA=0, shrinkB=0)
# parameter to increase arrow length because it's too short
add = 0.05
subtract = 0.05
#
sd_e_x_z = np.sqrt(cv_x-e_cv_x_z)
between_cl_sd_arrow = ax1.annotate('', xy=(x_c-sd_e_x_z*(1-p) -subtract,
                             (fake_1-fake_0)/2),
                             xytext=(x_c+sd_e_x_z*(p)+add,
                                     (fake_1-fake_0)/2),
                             arrowprops=arrow_properties)
## within-cluster standard deviation arrows
within_cl_sd_0= ax1.plot([], [], color=color2, lw=variance_lw)[0]
arrow_properties0 = dict(arrowstyle="<->", color=color2,
                         lw=variance_lw, shrinkA=0, shrinkB=0)
sd_x_z0 = np.sqrt(cv_x_z0)
within_cl_sd_0_arrow = ax1.annotate('', (e_x_z0-sd_x_z0/2-subtract,
                                  fake_0-scatter_shift_bigger),
                             (e_x_z0+sd_x_z0/2+add,
                              fake_0-scatter_shift_bigger),
                             arrowprops=arrow_properties0)
within_cl_sd_1 = ax1.plot([], [], color=color2, lw=variance_lw)[0]
arrow_properties1 = dict(arrowstyle="<->", color=color2,
                         lw=variance_lw, shrinkA=0, shrinkB=0)
sd_x_z1 = np.sqrt(cv_x_z1)
within_cl_sd_1_arrow = ax1.annotate('', (e_x_z1-sd_x_z1/2-subtract,
                                  fake_1-scatter_shift_bigger),
                             (e_x_z1+sd_x_z1/2+add,
                              fake_1-scatter_shift_bigger),
                             arrowprops=arrow_properties1)
## scatter plot of first j_plot simulations of the generic code
# first j_plot simulations of x and z
x_plot = x[:j_plot]
z_plot = z[:j_plot]
# number of simulations which have generic code 1 among first j_plot simulation
size_z1 = np.size(z_plot[z_plot])
# scatter plot of generic code
plot_x_z0 = ax1.scatter(x_plot[~z_plot], np.zeros(j_plot-size_z1)- scatter_shift,
                     facecolor='none',
                     color=light_green_2)
plot_x_z1 = ax1.scatter(x_plot[z_plot], np.ones(size_z1)- scatter_shift,
                     facecolor='none',
                     color=teal)
## vertical line representing threshold
cplot1 = ax1.plot([x_c, x_c], [ylim[0], ylim[1]], lw=0.5,
                  color=light_grey)[0]

### axes 2 - Distribution of generic code

ax2 = fig.add_axes(pos3, frameon=False)
ax2.text(x=0.15, y=fake_1*1.5, s='Pdf $Z$',# rotation=-90,
         fontdict=dict(fontweight='bold', fontsize=20))
ax2.tick_params(axis='y', labelleft=False, left=False)
ax2.set_xlim([0, 1])
ax2.set_ylim([-0.5, fake_1+0.58])
ax2.set_xticks([0, 0.5, 1])
ax2.set_yticks([fake_0, fake_1])
ax2.yaxis.set_label_position('right')
ax2.grid(False)
## bars representing Bernoulli Z
q_bar0 = ax2.barh(fake_0, 1-p, color=light_green_2)[0]
q_bar1 = ax2.barh(fake_1, p, color=teal, left=1-p)[0]
rect_width = q_bar0.get_height()
## line connecting 2 green bars
bar_line = ax2.plot([1-p, 1-p],
                    [fake_0+rect_width/2, fake_1-rect_width/2],
                    lw=0.5, c=light_green_2)[0]

### axes 3 - Optimal errors of autoencoders as a function of boundary points and error of the generic autoencoder

ax3 = fig.add_axes(pos2)
ax3.set_xlim(x_c_grid_lim)
ax3.set_ylim(0, cv_x+0.15)
ax3.tick_params(axis='y', right=False, labelright=False,
                labelleft=True, left=True)
ax3.set_xlabel('$x$', fontsize=20)
ax3.xaxis.set_label_coords(1, -0.0055)
ax3.grid(False)
## total variance
tot_var = ax3.plot(ax3.get_xlim(), [cv_x, cv_x], color=orange,
                   lw=variance_lw)[0]
## 2 orange areas representing within-cluster and between-cluster variances
# within-cluster variance as a function of boundary points
e_cv_x_z_grid = np.empty_like(x_c_grid)
for i in range(x_c_grid_size):
    z_i = x > x_c_grid[i]
    x_z1_i = x[z_i]
    x_z0_i = x[~ z_i]
    p_i = (np.size(x_z1_i))/j_
    if p_i==1 or p_i==0:
        e_cv_x_z_grid[i] = cv_x
    else:
        cv_x_z0_i = meancov_sp(x_z0_i)[1]
        cv_x_z1_i = meancov_sp(x_z1_i)[1]
        e_cv_x_z_grid[i] = (1-p_i)*cv_x_z0_i + p_i*cv_x_z1_i
ax3.fill_between(x_c_grid, e_cv_x_z_grid, cv_x, color=sand)
ax3.fill_between(x_c_grid, e_cv_x_z_grid, color=color2)
## vertical line representing error of the generic autoencoder
plot_e_x_c_x0_x1 = ax3.plot([x_c, x_c], [0, e_x_c_x0_x1],
                           lw=variance_lw, color=colf)[0]
## vertical line representing generic boundary point
cplot_right = ax3.plot([x_c, x_c], ax3.get_ylim(),
                       lw=0.5, color=light_grey)[0]


### axes 4 - Sumulations and pdf of X

ax4 = fig.add_axes(pos4)
ax4.set_xlim(x_c_grid_lim)
ax4.set_ylim(-0.1, 0.45)
ax4_ylim = ax4.get_ylim()
ax4.set_xlabel('$x$', fontsize=20)
ax4.xaxis.set_label_coords(1, -0.0055)
ax4.grid(False)
scaling = 2.8
# grid
no_grid = 600
grid = np.linspace(x_c_grid_lim[0], x_c_grid_lim[1]+0.2, no_grid)
# split the grid
grid0 = grid[grid <= x_c]
grid1 = grid[grid > x_c]
## pdf of X
def normal_pdf(x, mu=0, sigma2=1):
    return np.exp(-(x-mu)**2/(2*sigma2))/np.sqrt(2*np.pi*sigma2)
def pdf_x(x):
    return (1-p_h)*normal_pdf(x, mu_x_0, sigma2_x_0) +\
            p_h*normal_pdf(x, mu_x_1, sigma2_x_1)
ax4.plot(grid, multiplier*pdf_x(grid), lw=2, color=dark_grey)
## split pdf
fill0 = ax4.fill(np.r_[grid0, grid0[-1]],
                 np.r_[multiplier*pdf_x(grid0), 0],
                 color=light_green_2)[0]
fill1 = ax4.fill(np.r_[grid1, grid1[0]],
                 np.r_[multiplier*pdf_x(grid1), 0],
                 color=teal)[0]
## vertical line representing threshold
cplot = ax4.plot([x_c, x_c], [ax4_ylim[0]-0.5, ax4_ylim[1]+0.5],
                 lw=zeta_lw, color=colorc)[0]
## first j_plot simulations of x
ax4.scatter(x_plot, np.zeros(x_plot.shape)-scatter_shift/scaling,
            facecolor='none', color=color0)
## standard deviation of x arrow
arrow_properties = dict(arrowstyle="<->", color=orange,
                         lw=variance_lw, shrinkA=0, shrinkB=0)
sd_x=np.sqrt(cv_x)
ax4.annotate('', (e_x-sd_x/2-subtract, -scatter_shift_bigger/(scaling-0.2)),
             (e_x+sd_x/2+add, -scatter_shift_bigger/(scaling-0.2)),
             arrowprops=arrow_properties)
## arbitrary points
arb1 = ax4.plot([arb_point_0, arb_point_1],
                [0-multipp*scatter_shift/scaling,
                 0-multipp*scatter_shift/scaling], 'o',
                color=color0, markersize=markersize-3)[0]

### legend

dummy_black_line = mlines.Line2D([], [], color=dark_grey, lw=2)
dummy_conditional_0l = ax1.plot([], [], '-', color=light_green_2,
                                lw=2)[0]
dummy_conditional_0c = ax1.plot([], [], 'o', color=color0, markersize=6,
                                fillstyle='none')[0]
dummy_conditional_1l = ax1.plot([], [], '-', color=teal,
                                lw=2)[0]
dummy_conditional_1c = ax1.plot([], [], 'o', color=color0,
                                fillstyle='none')[0]
dummy_leg = []
dummy_labels = []
for i in range(8):
    if i == 0:
        dummy_leg.append(dummy_conditional_0c)
        dummy_labels.append('')
    else:
        dummy_leg.append(mlines.Line2D([], [], color='none'))
        dummy_labels.append('')
dummy_leg = tuple(dummy_leg)
dummy_labels = tuple(dummy_labels)
leg_loc = (0.71+hh, 0.031)
leg_vert_spacing = 2.21
fig.legend((dummy_black_line, cplot, plot_x0, plot_e_x_z0, plot_e_x_c_x0_x1, tot_var, between_cl_sd, within_cl_sd_0),\
           ('Scenarios/pdf $X$', '$z=\zeta(x)$ generic encoder',\
            '$\overline{x}=\chi(z)$ generic decoder',\
            '$\overline{x}=\mathbb{E}\{X|z\}$ best decoder', 'Error',\
            'Total variance', 'Between-cluster variance',\
            'Within-cluster variance'),\
           loc=leg_loc, prop={'size': '17', 'weight': 'bold'}, numpoints=1,\
           facecolor='none', edgecolor='none',\
           handletextpad=0.5, labelspacing=leg_vert_spacing)
fig.legend(dummy_leg,
           dummy_labels,
           loc=(leg_loc[0]-0.008, leg_loc[1]-0.017),
           prop={'size': '17', 'weight': 'bold'}, numpoints=1,
           facecolor='none', edgecolor='none',
           handlelength=0.1, handletextpad=0.2,
           labelspacing=leg_vert_spacing*2.15/2)

add_logo(fig, axis=ax1, set_fig_size=False, location=2)