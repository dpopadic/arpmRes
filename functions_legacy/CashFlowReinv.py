import matplotlib.pyplot as plt
import numpy as np
from numpy import cumprod, zeros
from numpy import sum as npsum

plt.style.use('seaborn')


def CashFlowReinv(t_to_u, t_k, ind_t_k, Reinv_0_dt, c):
    # This function computes the reinvested cash-flow stream for a coupon bond
    #
    # INPUTS:
    # t_to_u            [vector]: 1 x m_ calendar times
    # t_k               [vector]: 1 x k_ coupon payment dates such that r_k in [t,u)
    # ind_t_k           [vector]: indexes of t_to_u corresponding to the payment dates
    # Reinv_0_dt        [vector]: 1 x m_  reinvestment factors for step dt
    # c                 [vector]: 1 x k_ amount paid at each coupon payment (for unit of notional)
    #
    # OUTPUTS:
    # CF_u              [vector]: 1 x m_ reinvested cash-flow stream

    # ensure that variables have at least two dimensions
    t_to_u = np.atleast_2d(t_to_u)
    t_k = np.atleast_2d(t_k)
    Reinv_0_dt = np.atleast_2d(Reinv_0_dt)

    CF = zeros((t_k.shape[1], t_to_u.shape[1]))

    for k in range(t_k.shape[1] - 1):
        CF[k, ind_t_k[k] + 1:ind_t_k[-1] + 1] = c[k] * cumprod(Reinv_0_dt[0, ind_t_k[k] + 1:ind_t_k[-1] + 1])

    CF[t_k.shape[1] - 1, ind_t_k[-1]] = c[-1]
    CF_u = npsum(CF, 0)
    return CF_u

