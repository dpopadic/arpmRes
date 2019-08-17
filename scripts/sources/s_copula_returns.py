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

# # s_copula_returns [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_copula_returns&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-copula-comp-lin-ret).

# +
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_copula_returns-parameters)

mu = np.array([-3.2, 1.7])  # expectations
svec = np.array([0.003, 0.195])  # standard deviations
rho = 0.25  # correlation
sigma2 = np.diag(svec) @ np.array([[1, rho], [rho, 1]]) @ np.diag(svec)
j_ = 20000  # number of scenarios

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_copula_returns-implementation-step01): Generate scenarios for the bivariate normal compounded returns

c = simulate_normal(mu, sigma2, j_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_copula_returns-implementation-step02): Compute copula of the compounded returns

u1 = stats.norm.cdf(c[:, 0], mu[0], np.sqrt(sigma2[0, 0]))
u2 = stats.norm.cdf(c[:, 1], mu[1], np.sqrt(sigma2[1, 1]))
u_c = np.array([u1, u2]).T

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_copula_returns-implementation-step03): Map compounded returns into linear returns

r = np.exp(c) - 1

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_copula_returns-implementation-step04): Compute copula of the linear returns

u1 = stats.norm.cdf(np.log(r[:, 0]+1), mu[0], np.sqrt(sigma2[0, 0]))
u2 = stats.norm.cdf(np.log(r[:, 1]+1), mu[1], np.sqrt(sigma2[1, 1]))
u_r = np.array([u1, u2]).T

# ## Plots

# +
plt.style.use('arpm')

y_color = [153/255, 205/255, 129/255]
u_color = [60/255, 149/255, 145/255]

r1lim = [np.percentile(r[:, 0], 0.5), np.percentile(r[:, 0], 99.5)]
r2lim = [np.percentile(r[:, 1], 0.5), np.percentile(r[:, 1], 99.5)]

plt.figure()
mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
gs0 = gridspec.GridSpec(2, 2)
gs00 = gridspec.GridSpecFromSubplotSpec(46, 18, subplot_spec=gs0[0],
                                        wspace=0, hspace=0.6)
ax1 = plt.Subplot(f, gs00[:-10, 5:-5])
f.add_subplot(ax1)
plt.scatter(c[:, 0], c[:, 1], s=5, color=y_color)
ax1.tick_params(axis='x', colors='None')
ax1.tick_params(axis='y', colors='None')
plt.title('Compounded returns', fontsize=20, fontweight='bold')

ax11 = plt.Subplot(f, gs00[:-10, 2:4])
f.add_subplot(ax11)
plt.hist(np.sort(c[:, 1]), bins=int(30*np.log(j_)),
         orientation='horizontal', color=y_color, bottom=0)
ax11.tick_params(axis='x', colors='None')
plt.gca().invert_xaxis()

ax12 = plt.Subplot(f, gs00[40:46, 5:-5], sharex=ax1)
f.add_subplot(ax12)
plt.hist(np.sort(c[:, 0]), bins=int(120*np.log(j_)),
         color=y_color, bottom=0)
plt.gca().invert_yaxis()
ax12.tick_params(axis='y', colors='None')

gs01 = gridspec.GridSpecFromSubplotSpec(46, 18, subplot_spec=gs0[1],
                                        wspace=0, hspace=0.6)
ax2 = plt.Subplot(f, gs01[:-10, 5:-5], xlim=r1lim)
f.add_subplot(ax2)
plt.scatter(r[:, 0], r[:, 1], s=5, color=y_color)
ax2.tick_params(axis='x', colors='None')
ax2.tick_params(axis='y', colors='None')
plt.title('Linear returns', fontsize=20, fontweight='bold')

ax21 = plt.Subplot(f, gs01[:-10, 2:4])
f.add_subplot(ax21)
plt.hist(np.sort(r[:, 1]), bins=int(30*np.log(j_)),
         orientation='horizontal', color=y_color, bottom=0)
plt.gca().invert_xaxis()
ax21.tick_params(axis='x', colors='None')

ax22 = plt.Subplot(f, gs01[40:46, 5:-5], sharex=ax2)
f.add_subplot(ax22)
plt.hist(np.sort(r[:, 0]), bins=int(30*np.log(j_)),
         color=y_color, bottom=0)
plt.gca().invert_yaxis()
ax22.tick_params(axis='y', colors='None')

gs02 = gridspec.GridSpecFromSubplotSpec(46*2, 18*2,
            subplot_spec=gs0[1, :], wspace=0.6, hspace=1)
ax3 = plt.Subplot(f, gs02[:-10*2, 13:-14])
f.add_subplot(ax3)
plt.scatter(u_r[:, 0], u_r[:, 1], s=5, color=u_color)
plt.title('Copula', fontsize=20, fontweight='bold')
ax3.tick_params(axis='x', colors='None')
ax3.tick_params(axis='y', colors='None')

ax31 = plt.Subplot(f, gs02[:-10*2, 10:12])
f.add_subplot(ax31)
plt.hist(np.sort(u_r[:, 1]), bins=int(30*np.log(j_)),
         orientation='horizontal', color=u_color, bottom=0)
plt.gca().invert_xaxis()
ax31.tick_params(axis='x', colors='None')

ax32 = plt.Subplot(f, gs02[40*2:46*2, 13:-14])
f.add_subplot(ax32)
plt.hist(np.sort(u_r[:, 0]), bins=int(120*np.log(j_)),
         color=u_color, bottom=0)
plt.gca().invert_yaxis()
ax32.tick_params(axis='y', colors='None')

add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()
