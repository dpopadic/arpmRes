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

# # s_scen_prob_quantile [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_scen_prob_quantile&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-univ-fpcase-stud).

# +
import numpy as np

from arpym.statistics import quantile_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_quantile-parameters)

# +
k_ = 99

x = np.array([1, 2, 0])
p = np.array([0.31, 0.07, 0.62])
c_ = np.linspace(0.01, 0.99, k_)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_quantile-implementation-step01): Compute quantile

q_x_c = quantile_sp(c_, x, p)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_scen_prob_quantile-implementation-step02): Compute the median

med_x = quantile_sp(0.5, x, p)
