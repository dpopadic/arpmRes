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

# # s_logn_uncertainty_bands [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_logn_uncertainty_bands&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_logn_uncertainty_bands).

# +
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_logn_uncertainty_bands-parameters)

mu = 1  # parameters of lognormal
sigma2 = 0.5
r_1 = 1  # radius
r_2 = 1.5
x_grid = np.linspace(stats.lognorm.ppf(0.001, np.sqrt(sigma2), loc=mu),
                     stats.lognorm.ppf(0.999, np.sqrt(sigma2), loc=mu), 100)  # evaluation points

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_logn_uncertainty_bands-implementation-step01): Compute expectation, standard deviation, and uncertainty bands

exp_x = np.exp(mu + 0.5*sigma2)  # expectation
v_x = (exp_x**2) * (np.exp(sigma2)-1)
std_x = np.sqrt(v_x)  # standard deviation
u_x_r_1 = [exp_x-r_1*std_x, exp_x+r_1*std_x]  # uncertainty bands
u_x_r_2 = [exp_x-r_2*std_x, exp_x+r_2*std_x]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_logn_uncertainty_bands-implementation-step02): Compute the pdf for a grid of evaluation points

pdf = stats.lognorm.pdf(x_grid, np.sqrt(sigma2), loc=mu)

# ## Plots

# +
plt.style.use('arpm')
fig = plt.figure(figsize=(1280/72, 720/72), dpi=72)

# pdf
plt.plot(x_grid, pdf, 'k')

# uncertainty bands
plt.plot(np.linspace(min(u_x_r_1[0], u_x_r_2[0]),
                          max(u_x_r_1[1], u_x_r_2[1]), 10),
              np.zeros(10), 'g-', lw=5,
              label='Uncertainty band r = %.1f' %max(r_1, r_2))

plt.plot(np.linspace(max(u_x_r_1[0], u_x_r_2[0]),
                          min(u_x_r_1[1], u_x_r_2[1]), 10),
              np.zeros(10), 'b-', lw=7, label='Uncertainty band r = %.1f' %min(r_1, r_2))

# expectation
plt.plot(exp_x, 0, 'r.', ms=11, label='Expectation')

plt.legend(prop={'size': 17})
plt.title('Lognormal distribution: uncertainty bands', fontweight='bold', fontsize=20)
add_logo(fig, location=4, size_frac_x=1/16, set_fig_size=False)
plt.tight_layout()
