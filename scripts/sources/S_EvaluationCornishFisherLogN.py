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

# # S_EvaluationCornishFisherLogN [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_EvaluationCornishFisherLogN&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBCornishFisherEvaluation).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from PortfolioMomentsLogN import PortfolioMomentsLogN
from CornishFisher import CornishFisher

# parameters
v_tnow = array([[2], [1.5]])
mu = array([[0.5], [-0.3]])
sigma2 = array([[0.55, 0.82],
          [0.82, 1.05]])
h = array([[2], [1]])
c = 0.95
# -

# ## Computation of the expectation, the standard deviation and the skewness
# ## of the portfolio's P&L using function PortfolioMomentsLogN

muPL_h, sdPL_h, skPL_h = PortfolioMomentsLogN(v_tnow, h, mu, sigma2)

# ## Using the skewness computed at the previous step, compute the third central
# ## moment of the portfolio's P&L

third_central = skPL_h@(sdPL_h) ** 3

# ## Computation of the Cornish-Fisher expansion of the quantile based-index
# ## with confidence c=0.95 using function CornishFisher

q = CornishFisher(muPL_h, sdPL_h, skPL_h, 1 - c)
