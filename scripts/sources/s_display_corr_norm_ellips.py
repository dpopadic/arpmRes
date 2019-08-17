# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_display_corr_norm_ellips [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_display_corr_norm_ellips&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ellipso-norm-biv-var).

# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from arpym.statistics import simulate_normal
from arpym.tools import histogram_sp, pca_cov, plot_ellipse, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_display_corr_norm_ellips-parameters)

mu_x = np.array([0, 0])  # expectation
rho = 0.75  # correlation
sigma2_x = np.array([[1, rho],
                    [rho, 1]])  # covariance
r = 2  # radius
j_ = 10000  # number of scenarios

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_display_corr_norm_ellips-implementation-step01): Compute eigenvalue and eigenvectors

e, lambda2 = pca_cov(sigma2_x)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_display_corr_norm_ellips-implementation-step02): Compute simulations of the target and factors

x = simulate_normal(mu_x, sigma2_x, j_)
z = (x-mu_x) @ e

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_display_corr_norm_ellips-implementation-step03): Perform computations for plots

x_bar = plot_ellipse(mu_x, sigma2_x, r=r, display_ellipse=False, plot_axes=True, plot_tang_box=True,
             color='k')
[f_z1, xi_z1] = histogram_sp(z[:, 0], k_=300)
[f_z2, xi_z2] = histogram_sp(z[:, 1], k_=300)

# ## Plots

# +
plt.style.use('arpm')

mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)

# setup range
x_range = [-5, 5]

# long and short axis
u_axes0 = e @ (np.sqrt(lambda2) * np.array([[-r, r], [0, 0]]).T).T
u_axes1 = e @ (np.sqrt(lambda2) * np.array([[0, 0], [-r, r]]).T).T


# generate figure
f.text(0.1, 0.2, 'correlation = {rho:.2f}'.format(rho=rho))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[3, 1])

ax0 = plt.subplot(gs[0])
plt.barh(xi_z2, f_z2, facecolor=[.7, .7, .7])
ax0.set_ylim(x_range)
ax0.plot([0, 0], [-r*np.sqrt(lambda2[1]), r*np.sqrt(lambda2[1])], linewidth=5.0, color='blue')
ax0.set_ylabel('$Z_2^{PC}$', fontsize=14)

ax1 = plt.subplot(gs[1])
ax1.scatter(x[:, 0], x[:, 1], color=[.8, .8, .8], marker='.', s=40)
plot_ellipse(mu_x, sigma2_x, r=r, display_ellipse=True, plot_axes=False, plot_tang_box=True,
             color='k')
ax1.plot(u_axes0[0], u_axes0[1], linewidth=2.0, color='red')
ax1.plot(u_axes1[0], u_axes1[1], linewidth=2.0, color='blue')
ax1.set_xlim(x_range)
ax1.tick_params(axis='y', colors='None')
ax1.tick_params(axis='x', colors='None')
ax1.set_ylim(x_range)
ax1.set_xlabel('$X_1$', labelpad=-1, fontsize=12)
ax1.set_ylabel('$X_2$', labelpad=-20, fontsize=12)

ax2 = plt.subplot(gs[3])
plt.bar(xi_z1, f_z1, facecolor=[.7, .7, .7])
ax2.set_xlim(x_range)
ax2.plot([-r*np.sqrt(lambda2[0]), r*np.sqrt(lambda2[0])], [0, 0], linewidth=5.0, color='red')
ax2.set_xlabel('$Z_1^{PC}$', fontsize=14)

plt.tight_layout()
add_logo(f, location=1, size_frac_x=1/12)
# -


