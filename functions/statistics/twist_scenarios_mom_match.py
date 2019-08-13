# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.meancov_sp import meancov_sp
from arpym.tools.transpose_square_root import transpose_square_root


def twist_scenarios_mom_match(x, m_, s2_, p=None, method='Riccati', d=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (j_,n_) if n_>1 or (j_,) for n_=1
        m_ : array, shape (n_,)
        s2_ : array, shape (n_,n_)
        p : array, optional, shape (j_,)
        method : string, optional
        d : array, shape (k_, n_), optional

    Returns
    -------
        x : array, shape (j_, n_) if n_>1 or (j_,) for n_=1

    """

    if np.ndim(m_) == 0:
        m_ = np.reshape(m_, 1).copy()
    else:
        m_ = np.array(m_).copy()
    if np.ndim(s2_) == 0:
        s2_ = np.reshape(s2_, (1, 1))
    else:
        s2_ = np.array(s2_).copy()
    if len(x.shape) == 1:
        x = x.reshape(-1, 1).copy()

    if p is None:
        j_ = x.shape[0]
        p = np.ones(j_) / j_  # uniform probabilities as default value

    # Step 1. Original moments

    m_x, s2_x = meancov_sp(x, p)

    # Step 2. Transpose-square-root of s2_x

    r_x = transpose_square_root(s2_x, method, d)

    # Step 3. Transpose-square-root of s2_

    r_ = transpose_square_root(s2_, method, d)

    # Step 4. Twist matrix

    b = r_ @ np.linalg.inv(r_x)

    # Step 5. Shift vector

    a = m_.reshape(-1, 1) - b @ m_x.reshape(-1, 1)

    # Step 6. Twisted scenarios

    x_ = (a + b @ x.T).T

    return np.squeeze(x_)
