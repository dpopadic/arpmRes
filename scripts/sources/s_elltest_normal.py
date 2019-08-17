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

# # s_elltest_normal [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_elltest_normal&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerIIDtests).

# +
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from arpym.statistics import cop_marg_sep, \
                             invariance_test_ellipsoid, \
                             simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_normal-parameters)

t_ = 1000  # time series length
mu = 0  # expectation
sigma2 = 0.0625  # variance
l_ = 10  # lag for the ellipsoid test
conf_lev = 0.95  # confidence level

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_normal-implementation-step00): Generate normal simulations

epsi = simulate_normal(mu, sigma2, t_)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_normal-implementation-step01): Compute absolute values of normal simulations

epsi_abs = abs(epsi)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_normal-implementation-step02): Compute normalized absolute values

# grades of absolute values
epsi_abs_grade, *_ = cop_marg_sep(epsi_abs)
# normalized absolute values
epsiabs_tilde = st.norm.ppf(epsi_abs_grade).squeeze()

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_normal-implementation-step03): Ellipsoid test on normal simulations

plt.style.use('arpm')
name1 = 'Invariance test on normal simulations'
acf_epsi, conf_int = \
    invariance_test_ellipsoid(epsi, l_, conf_lev=conf_lev,
                              fit=0, r=1.8,
                              title=name1)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_normal-implementation-step04): Ellipsoid test on absolute value of normal simulations

plt.style.use('arpm')
name2 = 'Invariance test on absolute values of normal simulations'
acf_abs, conf_int = \
    invariance_test_ellipsoid(epsi_abs, l_, conf_lev=conf_lev,
                              fit=0, r=1.8,
                              title=name2,
                              bl=[-0.1], bu=[0.7])
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_normal-implementation-step05): Ellipsoid test on normalized absolute values

plt.style.use('arpm')
name3 = 'Invariance test on normalized absolute values'
acf_til, conf_int = \
    invariance_test_ellipsoid(epsiabs_tilde, l_,
                              conf_lev=conf_lev, fit=0,
                              r=1.8, title=name3)
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)
