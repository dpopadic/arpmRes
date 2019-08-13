from numpy import arange, diag, log, exp, tile
from numpy import sum as npsum


def PnlVolatility_ewma(Prices):
    # Exponentially weighted moving average volatility

    daily_return = Prices[:,1:]/ Prices[:,:-1] - 1

    [n_, t_]=daily_return.shape

    #calculate vol by using ewma weight
    Smoothing_longterm=90
    coeff=-log(2)/Smoothing_longterm

    #calculate ewma coeff for each day t
    daily_coeff=exp(-coeff*arange(t_))
    sumcoeff=npsum(daily_coeff)
    daily_weight=daily_coeff/sumcoeff

    #calculate cov_ewma and cov_pnl
    cov_ewma=(tile(daily_weight, (n_,1)) * daily_return )@daily_return.T
    cov_pnl=diag(Prices[:,-1])@cov_ewma@diag(Prices[:,-1])
    return cov_pnl
