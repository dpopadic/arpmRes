from collections import namedtuple

from numpy import std, zeros, cumsum, mean, r_, copy
from numpy import sum as npsum, min as npmin, max as npmax


def PnlStats(pnl_contr):
    # This function computes some Profit and loss (P&L) statistics
    #  INPUT
    #   pnl_contr   :[matrix] (n_ x t_end) portfolio P&L contributions
    #  OP
    #   stats       :[struct] fields: mean stdev sharpe_ratio mean_drawdown std_drawdown max_drawdown
    #   pnl_onestep :[vector] (1 x t_end) one-step P&L
    #   cumpnl      :[vector] (1 x t_end) cumulative P&L
    #   hwm         :[vector] (1 x t_end) high water mark
    #   dd          :[vector] (1 x t_end) drawdown

    # For details on the exercise, see here .

    ## Code
    t_ = pnl_contr.shape[1]

    pnl_onestep = npsum(pnl_contr, 0,keepdims=True) # one-step P&L
    cumpnl = cumsum(pnl_onestep,1) # cumulate pnl

    hwm = copy(cumpnl) # intialize
    for i in range(t_):
        hwm[0,i] = npmax(cumpnl[0,:i+1]) # high watermark

    dd = npmin(r_[cumpnl - hwm, zeros((0,t_))],0) # drawdown

    # statistics
    stats = namedtuple('stats',['mean','stdev','sharpe_ratio','mean_drawdown','std_drawdown','max_drawdown'])
    stats.mean = mean(pnl_onestep)
    stats.stdev = std(pnl_onestep)
    stats.sharpe_ratio = stats.mean/stats.stdev
    stats.mean_drawdown = mean(dd)
    stats.std_drawdown = std(dd)
    stats.max_drawdown = npmin(dd)
    return stats, pnl_onestep, cumpnl, hwm, dd
