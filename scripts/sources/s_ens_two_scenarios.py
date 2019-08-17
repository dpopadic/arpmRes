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

# # s_ens_two_scenarios [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_ens_two_scenarios&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerENScont).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.estimation import effective_num_scenarios
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_ens_two_scenarios-parameters)

k_ = 100  # size of grid of probabilities
min_p_1 = 0  # minimum value for p_1
max_p_1 = 1  # maximum value for p_1

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_ens_two_scenarios-implementation-step01): Create flexible probabilities scenarios

# create flexible probabilities
p_1 = np.linspace(min_p_1, max_p_1, num=k_)
p_2 = np.ones(k_)-p_1
p = np.vstack((p_1, p_2))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_ens_two_scenarios-implementation-step02): Calculate the effective number of scenarios

ens = np.zeros(k_)
for k in range(k_):
    ens[k] = effective_num_scenarios(p[:, k])

# ## Plots

# +
plt.style.use('arpm')

f = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.plot(p_1, ens, lw=1.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$p_1$', fontsize=17)
plt.ylabel('$ens(\mathbf{p})$', fontsize=17)
plt.title('Effective number of scenarios as the flexible probabilities vary\n'
          r'$\bar{t}=2$', fontsize=20, fontweight='bold')
add_logo(f, location=1, set_fig_size=False)
plt.tight_layout()
