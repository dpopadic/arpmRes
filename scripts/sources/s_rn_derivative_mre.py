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

# # s_rn_derivative_mre [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_rn_derivative_mre&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sdf-mre).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from arpym.statistics import simulate_normal, cdf_sp, pdf_sp
from arpym.pricing import numeraire_mre
from arpym.tools import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_rn_derivative_mre-implementation-step00): Upload data

# +

path = '../../../databases/temporary-databases/'

db_vpayoff = pd.read_csv(path+'db_valuation_vpayoff.csv', index_col=0)
v_payoff = db_vpayoff.values
db_vtnow = pd.read_csv(path+'db_valuation_vtnow.csv', index_col=0)
v_tnow = db_vtnow.values.T[0]
db_prob = pd.read_csv(path+'db_valuation_prob.csv', index_col=0)
p = db_prob.values.T[0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_rn_derivative_mre-implementation-step01): Compute the minimum relative entropy numeraire probabilities

# +
p_mre, sdf_mre = numeraire_mre(v_payoff, v_tnow, p=p, k=1)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_rn_derivative_mre-implementation-step02): Compute Radon-Nikodym derivative and inflator

# +
# compute Radon-Nikodym derivative
rnd_mre = p_mre / p
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_rn_derivative_mre-implementation-step03): Compute pdfs

# +
h = 0.02
# grid for computing pdfs
x = np.linspace(-1, 4, 100)
# compute pdfs
sdf_mre = pdf_sp(h, np.array([x]).T, np.array([sdf_mre]).T, p)
rnd_mre = pdf_sp(h, np.array([x]).T, np.array([rnd_mre]).T, p)
infl = pdf_sp(h, np.array([x]).T, np.array([v_payoff[:, 1]/v_tnow[1]]).T,
              p)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_rn_derivative_mre-implementation-step04): Compute cdfs under probability measures p and p_mre

# +
y = np.linspace(0, 12, 100)
ind = np.argsort(v_payoff[:, 1])
cdf = cdf_sp(y, v_payoff[:, 1], p)
cdf_mre = cdf_sp(y, v_payoff[:, 1], p_mre)
# -

# ## Plots

# +
plt.style.use('arpm')
sdf_name = r'$\mathit{SDF}_{t_{\mathit{now}}\rightarrow t_{\mathit{hor}}}^{\mathit{MRE}}$'
rnd_name = r'$\mathit{RND}_{t_{\mathit{now}}\rightarrow t_{\mathit{hor}}}^{\mathit{MRE}}$'
infl_name = r'$\mathit{V}_{2,t_{\mathit{now}}\rightarrow t_{\mathit{hor}}}^{\mathit{payoff}}/v_{2,t_{\mathit{now}}}$'

fig, axes = plt.subplots(1, 2)

axes[0].plot(x, sdf_mre, 'b', label=sdf_name)
axes[0].plot(x, rnd_mre, 'g', label=rnd_name)
axes[0].plot(x, infl, 'r', label=infl_name)
yl = axes[0].get_ylim()
axes[0].plot([v_tnow[0], v_tnow[0]], [0, yl[1]], 'b--',
             label=r'$E\{$' + sdf_name + '$\}$')
axes[0].plot([1, 1], [0, yl[1]], 'g--',
             label=r'$E\{$' + rnd_name + '$\}$')
axes[0].plot([p @ v_payoff[:, 1] / v_tnow[1],
              p @ v_payoff[:, 1] / v_tnow[1]], [0, yl[1]], 'r--',
             label=r'$E\{$' + infl_name + '$\}$')
axes[0].set_xlim([x[0], x[-1]])
axes[0].set_ylim(yl)
axes[0].legend()

axes[1].plot(y, cdf, 'b', label='$F$')
axes[1].plot(y, cdf_mre, 'g', label='$F^{numer}$')
axes[1].set_ylim([0, 1])
axes[1].set_xlabel(r'$\mathit{V}_{2,t_{\mathit{now}}\rightarrow t_{\mathit{hor}}}^{\mathit{payoff}}$')
axes[1].legend()

add_logo(fig, location=4, size_frac_x=1/8)
plt.tight_layout()
