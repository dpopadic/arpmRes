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

# # s_ens_exp_decay [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_ens_exp_decay&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBExpEntrProp2).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.estimation import effective_num_scenarios, exp_decay_fp
from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_ens_exp_decay-parameters)

gamma = 10  # parameter for the generalized exponential of entropy
t_ = 500  # number of scenarios
tau_hl_max = np.floor(1.2*t_)  # maximum half-life parameter
k_ = 50  # number of half-life parameters considered

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_ens_exp_decay-implementation-step01): Create a grid of half-life values for plotting

tau_hl_grid = np.linspace(1, tau_hl_max, num=k_)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_ens_exp_decay-implementation-step02): Compute exponential decay probabilities

p = np.zeros((k_, t_))
for k in range(k_):
    p[k] = exp_decay_fp(t_, tau_hl_grid[k])

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_ens_exp_decay-implementation-step03): Compute effective number of scenarios

ens = np.zeros(len(tau_hl_grid))
ens_gamma = np.zeros(k_)
for k in range(k_):
    ens[k] = effective_num_scenarios(p[k])
    ens_gamma[k] = effective_num_scenarios(p[k], type_ent='gen_exp',
                                           gamma=gamma)

# ## Plots

# +
plt.style.use('arpm')

f = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.xlim(0, np.ceil(tau_hl_max*10/t_)/10)
plt.ylim(0, 1)
plt.plot(tau_hl_grid/t_, ens/t_,
         label=r'$ens(\mathbf{p})\backslash \bar{t}$', linewidth=1.5)
plt.plot(tau_hl_grid/t_, ens_gamma/t_,
         label=r'$ens_{\gamma}(\mathbf{p})\backslash \bar{t}$', color='C3',
         linewidth=1.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\tau_{HL}\backslash \bar{t}$', fontsize=17)
plt.title('Effective number of scenarios for exponential decay probabilities',
          fontsize=20, fontweight='bold')
plt.legend(fontsize=17)
add_logo(f, location=4, set_fig_size=False)
plt.tight_layout()
