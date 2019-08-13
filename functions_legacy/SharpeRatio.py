from numpy import std, mean


def SharpeRatio(h,PL,r,v0):
    # This function computes the Sharpe ratio = (E[Pi_h]-r@v0)/Std[Pi_h] of a portfolio once we know
    # the risk-free rate r and the current budget v0.
    # INPUTS:
    # h     :[vector](n_ x 1) holdings
    # PL  : [matrix] (n_ x j_) scenarios for the P&L's of the n_ fizeroscial instruments in the portfolio
    # r   :[scalar] risk-free rate
    # v0   :[scalar] budget
    # OP:
    # ratio     :[scalar] Sharpe ratio

    Pi_h=h.T@PL
    ExcessPi_h=Pi_h-r@v0
    E=mean(ExcessPi_h)
    Sd=std(Pi_h)
    ratio=E/Sd
    return ratio
