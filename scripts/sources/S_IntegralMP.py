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

# # S_IntegralMP [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_IntegralMP&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=MarchenkoPasturIntegral).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from scipy.integrate import trapz

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from MarchenkoPastur import MarchenkoPastur

# initialize variable
q = 1.5
# -

# ## Compute the Marchenko-Pastur distribution

# +
sigma2 = 1
l_ = 100000

x, y, _ = MarchenkoPastur(q, l_, sigma2)
# -

# ## Compute the integral of the Marchenko-Pastur distribution

# +
if q >= 1:
    I = trapz(y, x)
else:
    I = trapz(y[1:l_], x[1:l_]) + (1 - q)  # accounting for the Dirac delta when q<1

print(I)
