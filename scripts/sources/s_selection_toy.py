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

# # s_selection_toy [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_selection_toy&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_selection_toy).

# +
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

from arpym.tools import naive_selection, forward_selection, \
    backward_selection

# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_selection_toy-parameters)

q2 = np.diag(np.array([1.1, 0.6, 1.2]))  # quadratic term of the obj. function
l = np.array([1.2, 2.1, -3.2])  # linear term of the obj. function
comb_nk = np.array([1, 2, 3])  # pool of candidates
k_ = 2  # maximum cardinality of a selection via selection routines


# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_selection_toy-implementation-step01): Performance function

def g(s_k):
    # define quadratic objective
    q2_ = matrix(2 * q2, tc='d')
    l_ = matrix(-l.reshape(-1, 1), tc='d')

    # define constraints
    constraints_f_s_lhs = np.ones((1, 3))
    constraints_f_s_rhs = np.ones((1, 1))
    if s_k.shape[0] <= 2 and s_k.shape[0] > 0:
        idt = np.eye(3)
        not_in_s_k = np.array([j - 1 for j in comb_nk
                               if j not in s_k])
        constraints_f_s_lhs[0, not_in_s_k] = 0
        constraints_f_s_lhs = np.concatenate((constraints_f_s_lhs, idt[not_in_s_k, :]))
        constraints_f_s_rhs = np.concatenate((constraints_f_s_rhs,
                                              np.zeros((not_in_s_k.shape[0], 1))))
    constraints_f_s_lhs = matrix(constraints_f_s_lhs, tc='d')
    constraints_f_s_rhs = matrix(constraints_f_s_rhs, tc='d')

    # solve optimization
    solvers.options['show_progress'] = False
    sol = solvers.qp(q2_, l_, A=constraints_f_s_lhs, b=constraints_f_s_rhs)
    x_star = np.array(sol['x'])
    g = -x_star.T @ q2 @ x_star + l @ x_star

    return np.asscalar(g)


# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_selection_toy-implementation-step02): Best selection of 1, 2 and 3 elements according to the performance g

# +
comb_31 = np.array([[1], [2], [3]])
comb_32 = np.array([[1, 2], [1, 3], [2, 3]])
comb_33 = np.array([[1, 2, 3]])

s_star = []
g_s_star = np.ones(3) * np.nan
s_star.append(comb_31[np.argmax([g(comb_31[0]),
                                 g(comb_31[1]), g(comb_31[2])])])
g_s_star[0] = g(s_star[0])

s_star.append(comb_32[np.argmax([g(comb_32[0]),
                                 g(comb_32[1]), g(comb_32[2])])])
g_s_star[1] = g(s_star[1])
g2 = [g(comb_32[0]), g(comb_32[1]), g(comb_32[2])]

s_star.append(comb_33[0])
g_s_star[2] = g(s_star[2])
g1 = [g(s1) for s1 in comb_31]
g2 = [g(s2) for s2 in comb_32]
g3 = [g(s3) for s3 in comb_33]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_selection_toy-implementation-step03): Optimal number of elements using the naive selection routine

# +
n_ = comb_nk.size

s_star_naive = naive_selection(g, n_, k_)
g_s_star_naive = [g(s_star_naive[k]) for k in range(k_)]
k_star_naive = k_
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_selection_toy-implementation-step04): Optimal number of elements using the forward selection routine

s_star_fwd = forward_selection(g, n_, k_)
g_s_star_fwd = [g(s_star_fwd[k]) for k in range(k_)]
k_star_fwd = k_

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_selection_toy-implementation-step05): Optimal number of elements using the backward selection routine

s_star_bwd = backward_selection(g, n_, k_)
g_s_star_bwd = [g(s_star_bwd[k]) for k in range(k_)]
k_star_bwd = k_
