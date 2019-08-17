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

# # S_PricingCarryZcb [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingCarryZcb&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExCarryNelsSieg).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import exp

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# parameters

# Nelson-Siegel parameters for the forward/yield curve
theta1 = 0.05
theta2 = 0.01
theta3 = 0.04
theta4 = 0.5

tau1 = 0.5  # time to maturity at time t
tau2 = 0.49  # time to maturity at time t+deltat
# -

# ## Computation of the exact annualized zero-coupon bond carry return

y_NS =lambda x: theta1 - theta2*(1 - exp(-theta4 ** 2*x)) / (theta4 ** 2*x) + theta3*(
(1 - exp(-theta4 ** 2*x)) / (theta4 ** 2*x) - exp(-theta4 ** 2*x))  # spot yield curve according to the Nelson-Siegel parametrization
exact_carry = (exp(tau1*y_NS(tau1) - tau2*y_NS(tau2)) - 1) / (tau1 - tau2)

# ## Computation of the approximated zero-coupon bond carry return

# +
der_yNS =lambda x: - (theta2 / ((theta4*x) ** 2))*(exp((-(theta4 ** 2))*x)*(theta4 ** 2*x + 1) - 1) + theta3*(
(exp((-(theta4 ** 2))*x)*(theta4 ** 2*x + 1) - 1) / (theta4*x)**2 + theta4**2*exp(-(theta4)**2*x))

yield_income = y_NS(tau2)
roll_down = tau2*der_yNS(tau2)
approx_carry = yield_income + roll_down
