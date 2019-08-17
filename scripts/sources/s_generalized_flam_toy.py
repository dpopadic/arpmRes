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

# # s_generalized_flam_toy [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_generalized_flam_toy&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy).

# +
import numpy as np

from arpym.statistics import objective_r2, simulate_normal
from arpym.tools import solve_riccati
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-parameters)

# +
sig2 = np.array
rho = 0.3
epsi = 0.45
s = np.array([[0.3], [0.1]])
w = np.array([[1], [-3]])
sig = 1

sig2 = np.array([[1, 0.5, epsi, epsi],
                 [0.5, 1, epsi, epsi],
                 [epsi, epsi, 1, rho],
                 [epsi, epsi, rho, 1]])
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step01): conditional expectation and covariance

# +

def cond_exp_x(s, k=2, sig2=sig2):
    return sig2[:2, -k:] @ np.linalg.solve(sig2[-k:, -k:], s)


def cond_cov_x(k=2, sig2=sig2):
    return sig2[:2, :2] - sig2[:2, -k:] @ np.linalg.solve(sig2[-k:, -k:],
                                                          sig2[:2, -k:].T)
cond_mu_x = cond_exp_x(s)
cond_sig2_x = cond_cov_x()
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step02): Max of cond. info ratio and combination at which is attained

w_sig = sig * np.linalg.solve(cond_sig2_x, cond_mu_x) / \
              np.sqrt(cond_mu_x.T @ np.linalg.solve(cond_sig2_x, cond_mu_x))
max_ir = w_sig.T @ cond_mu_x / np.sqrt(w_sig.T @ cond_sig2_x @ w_sig)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step03): Max of cond. info ratio via flam and transfer coefficient

max_ir_flam = np.sqrt(cond_mu_x.T @ np.linalg.solve(cond_sig2_x,
                                                    cond_mu_x))
ir_arb = w.T @ cond_mu_x / np.sqrt(w.T @ cond_sig2_x @ w)
tc = ir_arb / max_ir_flam

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step04): Max. unconditional info ratios

# +

def uncond_max_ir(k, sig2=sig2):

    # Monte Carlo scenarios for the signals
    s_j = simulate_normal(np.zeros((2)), sig2[-2:, -2:], 1000).T
    cond_mu_x_j = cond_exp_x(s_j[:k, :], k, sig2)

    # Monte Carlo scenarios for the conditioned max info ratio
    max_ir_j = cond_mu_x_j.T @ \
        np.linalg.solve(cond_cov_x(k, sig2),
                        cond_mu_x_j)

    return np.sqrt(np.trace(max_ir_j) / 1000)

uncond_maxir_12 = uncond_max_ir(2)
uncond_maxir_1 = uncond_max_ir(1)
uncond_maxir_2 = uncond_max_ir(1)
print(uncond_maxir_12**2 - (uncond_maxir_1**2 + uncond_maxir_2**2))

# verify that (epsi << 1) implies weak signals
sig2_weak = np.array([[1, 0.5, 0.1, 0.1],
                     [0.5, 1, 0.1, 0.1],
                     [0.1, 0.1, 1, rho],
                     [0.1, 0.1, rho, 1]])
print(cond_cov_x(2, sig2_weak))
print(sig2[:2, :2])

# independent signals (rho = 0) and weak correlation (epsi << 1)
sig2_weak_ind = np.array([[1, 0.5, 0.1, 0.1],
                          [0.5, 1, 0.1, 0.1],
                          [0.1, 0.1, 1, 0],
                          [0.1, 0.1, 0, 1]])
maxir_12_weak_ind = uncond_max_ir(2, sig2_weak_ind)
maxir1_weak_ind = uncond_max_ir(1, sig2_weak_ind)
maxir2_weak_ind = uncond_max_ir(1, sig2_weak_ind)
print(maxir_12_weak_ind**2 - (maxir1_weak_ind**2 +
                              maxir2_weak_ind**2))
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step05): information coefficients

# +

def ic(k, sig2=sig2):
    return np.sqrt(2 * objective_r2(np.arange(k), sig2, 2, sig2[:2, :2]))

ic_12 = ic(2)
ic_1 = ic(1)
ic_2 = ic(1)
print(ic_12**2 - (ic_1**2 + ic_2**2))

# independent signals (rho = 0)
sig2_ind = np.array([[1, 0.5, epsi, epsi],
                     [0.5, 1, epsi, epsi],
                     [epsi, epsi, 1, 0],
                     [epsi, epsi, 0, 1]])
ic_12_ind = ic(2, sig2_ind)
ic_1_ind = ic(1, sig2_ind)
ic_2_ind = ic(1, sig2_ind)
print(ic_12_ind**2 - (ic_1_ind**2 + ic_2_ind**2))
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step06): linkage matrix

# +

def linkage(sig2=sig2):
    return np.linalg.solve(solve_riccati(sig2[:2, :2]),
                           np.linalg.solve(solve_riccati(sig2[2:, 2:]).T,
                                           sig2[:2, 2:].T).T)

p_s_x = linkage(sig2)
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step07): Fundamental law of active management (weak signals)

# +
sig2_weak = np.array([[1, 0.5, 0.1, 0.1],
                      [0.5, 1, 0.1, 0.1],
                      [0.1, 0.1, 1, rho],
                      [0.1, 0.1, rho, 1]])
p_s_x_weak = linkage(sig2_weak_ind)

# information coefficient
ic_linkage = np.sqrt(np.trace(p_s_x_weak @ p_s_x_weak.T))

# max information ratio
s_tilde = np.linalg.solve(solve_riccati(sig2_weak[2:, 2:]), s)
maxir_linkage = uncond_max_ir(2, sig2=sig2_weak)
print(maxir_linkage**2 - ic_linkage**2)
# -

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step08): Fundamental law of active management (weak and ind. signals)

# +
p_s_x_weak_ind = linkage(sig2_weak_ind)

# information coefficient (single signal)
ic_linkage_1 = np.sqrt(np.trace(p_s_x_weak[:, [0]] @ p_s_x_weak[:, [0]].T))
print(ic_linkage_1 * np.sqrt(2) - maxir_12_weak_ind)
