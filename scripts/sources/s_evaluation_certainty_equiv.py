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

# # s_evaluation_certainty_equiv [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_evaluation_certainty_equiv&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBcertequivexputilfun).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.tools import histogram_sp
from arpym.statistics import simulate_normal
from arpym.tools import solve_riccati, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_certainty_equiv-parameters)

j_ = 10**5  # number of scenarios
v_tnow = np.array([1, 1])  # current values
mu = np.array([0, 0])  # instruments P&L's expectations
h = np.array([45, 55])  # portfolio holdings
lambda_ = np.array([1 / 150, 1 / 200, 1 / 300])  # risk aversion parameters
rho = -0.5  # correlation parameter
# standard deviations appearing in the P&L's distributions
sig_11, sig_22 = 0.1, 0.3
sig2 = np.array([[(sig_11) ** 2, rho*sig_11*sig_22],
                [rho*sig_11*sig_22, (sig_22) ** 2]])

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_certainty_equiv-implementation-step01): Simulate j_ scenarios for the instruments P&L's

n_ = len(h)  # number of the instruments
# scenarios for the standard normal random variable Z
z = simulate_normal(np.zeros(n_), np.eye(n_), j_, 'PCA')
sigma_riccati = solve_riccati(sig2, np.eye(n_))  # Riccati root of sigma2
mu = np.array([mu]*j_)  # duplicate expectation for j_ scenarios
v_tnow = np.array([v_tnow]*j_)  # duplicate initial values for j_scenarios
pi = np.exp(mu + z@sigma_riccati) - v_tnow  # P&L's scenarios
p = np.ones(j_) / j_   # flat scenario-probabilities

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_certainty_equiv-implementation-step02): Compute the ex-ante performance scenarios

y_h = h@pi.T  # ex-ante performance scenarios
# number of bins for the ex-ante performance histogram
bins = np.round(150 * np.log(j_))
# centers and heights of the bins
heights, centers = histogram_sp(y_h, p=p, k_=bins)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_certainty_equiv-implementation-step03): Compute the certainty-equivalent

# +
# Define the utility function and its inverse


def utility_exp(y, lam):  # exponential utility function
    return -np.exp(-lam * y)


def ce_exp(z, lam):  # inverse function
    return - np.log(-z) / lam


# Compute the expected utility and the certainty-equivalent for three values
# of the risk aversion parameter lambda
expect_utility = np.array([])  # initialization of the expected utility
for lam in lambda_:
    expect_utility = np.append(expect_utility, p@utility_exp(y_h, lam))

# compute the certainty-equivalent
cert_eq = ce_exp(expect_utility, lambda_)
# -

# ## Plots

# +
# Plot the histogram of the ex-ante performance, together
# with the the certainty-equivalent associated to the three values of
# lambda.

plt.style.use('arpm')

fig = plt.figure()
# colors
gray = [.9, .9, .9]
color1 = [0.95, 0.35, 0]
color2 = [.3, .8, .8]
color3 = [.9, .7, .5]

heights_ = np.r_[heights[np.newaxis, :],
                 heights[np.newaxis, :]] / np.max(heights)
heights_[0, centers <= 0] = 0
heights_[1, centers > 0] = 0
width = centers[1] - centers[0]

b = plt.bar(centers, heights_[0], width=width,
            facecolor=gray, edgecolor=color2)
b = plt.bar(centers, heights_[1], width=width,
            facecolor=gray, edgecolor=color3)
p1 = plt.plot([cert_eq[0], cert_eq[0]], [0, 0], color=color1, marker='.',
              markersize=8)
p2 = plt.plot([cert_eq[1], cert_eq[1]], [0, 0], color='b', marker='.',
              markersize=8)
p3 = plt.plot([cert_eq[2], cert_eq[2]], [0, 0], color='k', marker='.',
              markersize=8)
plt.legend(['$\lambda$ = ' +
            str(round(lambda_[0], 4)) +
            ' high risk aversion ', '$\lambda$ = ' +
            str(round(lambda_[1], 4)) +
            ' medium risk aversion ', '$\lambda$ = ' +
            str(round(lambda_[2], 4)) +
            ' low risk aversion '])
plt.ylim([-0.05, 1.05])
plt.ylabel('Certainty-equivalent ($)')
plt.xlabel('Portfolio P&L ($)')
plt.title(r'Market ex-ante P&L distribution ($\rho$=' +
          str(rho) + ', $\sigma$=' + str(sig_11) + ', '
          + str(sig_22) + ')')
add_logo(fig, location=5)
