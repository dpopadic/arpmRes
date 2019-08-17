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

# # S_EvaluationMLPMnoCom [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EvaluationMLPMnoCom&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBCompMLPMnoCom).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, mean, exp, sqrt

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from FPmeancov import FPmeancov

# parameters
Y1 = array([[1],[0], [- 1]])
Y2 = exp(Y1)  # Y1 and Y2 are co-monotonic
p = array([[1,1,1]]).T / 3
# -

# ## Compute the mean-lower partial moment trade-offs of Y1, Y2 and Y1+Y2.
# ## The expectations of Y1, Y2 and Y1+Y2 are obtained using function
# ## FPmeancov

# +
Y1_ = Y1 - FPmeancov(Y1.T,p)[0]
Y2_ = Y2 - FPmeancov(Y2.T,p)[0]
Y12_ = (Y1 + Y2) - FPmeancov(Y1.T+Y2.T,p)[0]

mlpm_Y1 = mean(Y1) - sqrt(((Y1_ ** 2) * (Y1_ < 0)).T@p)  # mean-lower partial moment trade-off of Y1
mlpm_Y2 = mean(Y2) - sqrt(((Y2_ ** 2) * (Y2_ < 0)).T@p)  # mean-lower partial moment trade-off of Y2
mlpm_Ysum = mlpm_Y1 + mlpm_Y2  # sum of the two mean-lower partial moment trade-offs
mlpm_Y12 = mean(Y1 + Y2) - sqrt((((Y12_) ** 2) * (Y12_ < 0)).T@p)  # mean-lower partial moment trade-off of Y1+Y2
