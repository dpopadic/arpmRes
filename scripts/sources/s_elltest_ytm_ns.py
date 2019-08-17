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

# # s_elltest_ytm_ns [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_elltest_ytm_ns&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerNSiid).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics import invariance_test_ellipsoid
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_ns-parameters)

# +
l_ = 10  # lag for the ellipsoid test
conf_lev = 0.95  # confidence level
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_ns-implementation-step00): Load data

# +
df_data = pd.read_csv('../../../databases/temporary-databases/db_fit_yield_ns.csv')
theta = df_data.values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_ns-implementation-step01): Compute increments NS aprameters

# +
delta_theta = np.diff(theta, axis=0)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_ns-implementation-step02): Ellipsoid tests

# +
plt.style.use('arpm')

# names of figures
name = {}
name[0]=r'Invariance test(increments of level parameter $\theta_1$)'
name[1]=r'Invariance test(increments of slope parameter $\theta_2$)'
name[2]=r'Invariance test(increments of curvature parameter $\theta_3$)'
name[3]=r'Invariance test(increments of decay parameter $\theta_4^2$)'

acf = np.zeros((4, l_))
conf_int_x = np.zeros((4, 2))
# perform and show ellipsoid test for invariance on NS parameters
for k in range(4):
    acf[k, :], conf_int_x[k, :] = invariance_test_ellipsoid(delta_theta[:, k], l_, conf_lev=conf_lev, title=name[k])
    fig = plt.gcf()
    add_logo(fig, set_fig_size=False, size_frac_x=1/8)
