# -*- coding: utf-8 -*-
import numpy as np


def effective_num_scenarios(p, type_ent=None, gamma=None):
    """For details, see here.

    Parameters
    ----------
        p : array, shape (t_,)
        type_ent : string, optional
        gamma : scalar, optional

    Returns
    -------
        ens : scalar

    """

    if type_ent is None:
        type_ent = 'exp'

    if (type_ent == 'gen_exp') and (gamma is None):
        raise ValueError('Provide parameter of the generalized exponential')

    if type_ent == 'exp':
        p_nonzero = p[p>0]  # avoid log(0) in ens computation
        ens = np.exp(-p_nonzero@np.log(p_nonzero))
    else:
        ens = np.sum(p ** gamma) ** (-1 / (gamma - 1))

    return ens
