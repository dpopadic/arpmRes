# -*- coding: utf-8 -*-

import numpy as np

from arpym.views.min_rel_entropy_sp import min_rel_entropy_sp


def twist_prob_mom_match(x, m_, s2_=None, p=None):
    """For details, see here.

    Parameters
    ----------
        x : array, shape (j_,n_) if n_>1 or (j_,) for n_=1
        m_ : array, shape (n_,)
        s2_ : array, shape (n_,n_), optional
        p : array, optional, shape (j_,), optional

    Returns
    -------
        p_ : array, shape (j_,)
    """

    if np.ndim(m_) == 0:
        m_ = np.reshape(m_, 1).copy()
    else:
        m_ = np.array(m_).copy()
    if s2_ is not None:
        if np.ndim(s2_) == 0:
            s2_ = np.reshape(s2_, (1, 1))
        else:
            s2_ = np.array(s2_).copy()

    if len(x.shape) == 1:
        x = x.reshape(-1, 1).copy()
    j_, n_ = x.shape
    if p is None:
        p = np.ones(j_) / j_

    if n_ + (n_ * (n_ + 1)) / 2 > j_:
        print('Error!')

    # Step 1: Compute the equality constraints

    z_eq = x.T.copy()
    mu_view_eq = m_.copy()

    if s2_ is not None:
        s2_ = np.array(s2_).copy()
        s2 = s2_ + np.outer(m_, m_)
        x_t = x.T.copy()
        for n in range(n_):
            z_eq = np.vstack((z_eq, x_t[n:, :] * x_t[n]))
            mu_view_eq = np.r_[mu_view_eq, s2[n, n:]]

    # Step 2: Minimize the relative entropy

    p_ = min_rel_entropy_sp(p, z_eq=z_eq, mu_view_eq=mu_view_eq)

    return np.squeeze(p_)
