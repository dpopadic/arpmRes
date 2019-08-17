#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_norm_const_sw [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_norm_const_sw&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=NormalizConstSWmeas).

import numpy as np
from scipy.integrate import dblquad

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_norm_const_sw-parameters)

k = 12    # SW normalization constant

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_norm_const_sw-implementation-step01): Define the integrand functions

# +
def f1(u1, u2):
    return (abs((np.minimum(u1, u2) - u1 * u2)))


def f2(u1, u2):
    return (abs((np.maximum(u1 + u2 - 1, 0) - u1 * u2)))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_norm_const_sw-implementation-step02): Compute the integrals

# +
d1, _ = dblquad(f1, 0, 1, lambda x: 0, lambda x: 1)


d2, _ = dblquad(f2, 0, 1, lambda x: 0, lambda x: 1)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_norm_const_sw-implementation-step03): Compare the results with the SW normalization constant

print(1 / d1 - k)
print(1 / d2 - k)
