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

# # s_generalized_expon_entropy [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_generalized_expon_entropy&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBGenExpConv).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.estimation import effective_num_scenarios
from arpym.statistics import simulate_normal
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_expon_entropy-parameters)

t_ = 10000  # number of scenarios for flexible probabilities
k_ = 100  # number of values of gamma

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_expon_entropy-implementation-step01): Generate a random vector of flexible probabilities

# generate a vector of positive values
p = np.abs(simulate_normal(0, 1, t_))
p = p/np.sum(p)  # rescale so the probabilities add to one

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_expon_entropy-implementation-step02): Create a grid of gamma values

gamma_grid = np.linspace(0, 1-1.0e-7, num=k_)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_expon_entropy-implementation-step03): Calculate the effective number of scenarios

ens = effective_num_scenarios(p)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_expon_entropy-implementation-step04): Calculate the generalized effective number of scenarios for various values of gamma

ens_gamma = np.zeros(k_)
for k in range(k_):
    ens_gamma[k] = effective_num_scenarios(p, type_ent='gen_exp',
                                           gamma=gamma_grid[k])

# ## Plots

# +
plt.style.use('arpm')

f = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.plot(gamma_grid, ens_gamma, linewidth=1.5)
plt.axhline(y=ens, color='lightgray', linewidth=1.5)
plt.xlim((0, 1.1))
plt.ylim((np.floor(ens*0.95/500)*500, t_))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$\gamma$', fontsize=17)
plt.ylabel('$\mathit{ens}_{\gamma}(\mathbf{p})$', fontsize=17)
plt.title('Generalized exponential of entropy convergence',
          fontsize=20, fontweight='bold')
plt.legend(['Gen. exponential of entropy', 'Exponential of entropy'],
           fontsize=17)
plt.grid(False)
add_logo(f, location=3, set_fig_size=False)
plt.tight_layout()
