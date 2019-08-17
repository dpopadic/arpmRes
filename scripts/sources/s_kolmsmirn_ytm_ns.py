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

# # s_kolmsmirn_ytm_ns [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_kolmsmirn_ytm_ns&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-nsiid-copy-1).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics import invariance_test_ks
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_ytm_ns-parameters)

# +
conf_lev = 0.95  # confidence level
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_ytm_ns-implementation-step00): Load data

# +
df_data = pd.read_csv('../../../databases/temporary-databases/db_fit_yield_ns.csv')
theta = df_data.values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_ytm_ns-implementation-step01): Compute increments of realized NS parameters

# +
delta_theta = np.diff(theta, axis=0)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_kolmsmirn_ytm_ns-implementation-step02): Kolmogorov-Smirnov tests

# +
plt.style.use('arpm')

# names of figures
name = {}
name[0]=r'Kolmogorov-Smirnov test(increments of level parameter $\theta_1$)'
name[1]=r'Kolmogorov-Smirnov test(increments of slope parameter $\theta_2$)'
name[2]=r'Kolmogorov-Smirnov test(increments of curvature parameter $\theta_3$)'
name[3]=r'Kolmogorov-Smirnov test(increments of decay parameter $\theta_4^2$)'

z_ks = np.zeros(4)
z = np.zeros(4)
# perform and show ellipsoid test for invariance on NS parameters
for k in range(4):
    fig = plt.figure()
    z_ks, z = invariance_test_ks(delta_theta[:, k], conf_lev=conf_lev, title=name[k])
    fig_ks = plt.gcf()
    fig_ks.set_dpi(72.0)
    fig_ks.set_size_inches(1280.0/72.0, 720.0/72.0)
    add_logo(fig_ks, set_fig_size=False)
