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

# # S_CointVar1CaseStudy [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_CointVar1CaseStudy&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=excointvarone).

# ## Prepare the environment

# +
import os.path as path
import sys

from numpy.linalg import inv

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import eye

from numpy import array
from numpy.random import randn

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from ARPM_utils import rref

d_ = 3

b = array([[0.709135409950543,  -0.628067758718625,  -0.0139342224380737],
    [-0.0217901643837038,  1.06811189176670,    0.0501791372949480],
    [0.274350156367857,  -0.779920404936362,   0.402752698282761]])

b_minus_I = b - eye(d_)

bmI_ech = rref(b_minus_I)

d = b_minus_I[:,:2]

c = bmI_ech[:2,:]

c = c.T

q = randn(d_-1,d_-1)

d_tilde = d@q
c_tilde = c@inv(q.T)

sigma2_Y = c.T@sigma2_X@c

sigma2_Y_tilde = c_tilde.T@sigma2_X@c_tilde
