#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script computes the eigenvalues/eigenvectors decomposition of the
# transition matrix theta of a tri-variate Ornestein-Uhlenbeck process.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-trajectory-animation).

# +
# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import imag, array, real
from numpy.linalg import eig, pinv

from scipy.linalg import block_diag

import matplotlib.pyplot as plt

plt.style.use('seaborn')

theta = array([[-10 ** -5, -120, -10], [-120, 10, 210], [-10, -210, 10]])  # transition matrix

lam, beta = eig(theta)  # eigenvectors and eigenvalues of theta
alpha = real(beta) - imag(beta)  # real matrix of eigenvectors

# real diagonal-block matrix
gamma_j = lam[0]
gamma_ja = real(lam[1])
gamma_jb = imag(lam[1])
gamma = block_diag(gamma_j, array([[gamma_ja, gamma_jb], [-gamma_jb, gamma_ja]]))

# check theta
theta_check = alpha@gamma.dot(pinv(alpha))
