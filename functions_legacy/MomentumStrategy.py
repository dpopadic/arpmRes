from numpy import arange, std, zeros, argsort, mean, log, exp, sqrt
from numpy import min as npmin

from PnlVolatility_ewma import PnlVolatility_ewma


def MomentumStrategy(Prices,Parameters):
    # [h_target]=Strategy(Prices,Parameters)
    #INPUT
    # Prices     :[matrix] (n_ x t_end)
    # Parameters :[struc]
    #             Parameters.Rebalance        :[scalar]
    #             Parameters.TransactionCosts :[scalar]
    #             Parameters.Smoothing        :[scalar]
    #             Parameters.Scoring          :[scalar]
    #             Parameters.DailyVol         :[scalar]
    # OP
    # h_target   :[vector] (n_ x 1)  holdings
    # est_pnl_vol_daily :[scalar]    estimated volatility of the pnl
    #################################################################

    daily_return = Prices[:,1:]/Prices[:,:-1] - 1

    #ewma
    #[ewma]=calc_ewma_today(Parameters.Smoothing, daily_return)

    #z-score of ewma
    zscore=calc_zscore_today(Parameters, daily_return)

    #strategy - assign weight
    #[weight]=assign_weight(ewma, -1)
    weight=assign_weight(zscore, -1)

    #calculate units to buy
    [h_target, est_pnl_vol_daily]=calc_units(weight, Prices, Parameters.DailyVol)

    return h_target, est_pnl_vol_daily


def calc_zscore_today(Parameters, data_in):
    #[z]=calc_zscore_today(Parameters, Prices)

    ewma=calc_ewma(Parameters.Smoothing, data_in)
    #[n_,l_]=ewma.shape

    Day_zscore=Parameters.Scoring

    #calculate ewma for the days afterward
    average=mean(ewma[:,-1-Day_zscore+1:],2)
    stdev=std(ewma[:,-1-Day_zscore+1:],1,2)
    z=(ewma[:,-1]-average)/stdev
    return z


def calc_ewma(Smoothing, data_in):

    t_=data_in.shape[1]

    #ewma
    coeff=-log(2)/Smoothing
    Days_for_ewma=npmin(t_,Smoothing*10)

    #calculate ewma for each day t by using data_in from(t - Days_for_ewma + 1) to t
    sumcoeff=0
    for i in range(Days_for_ewma):
        sumcoeff=1+exp(coeff)@sumcoeff

    ewma = zeros((data_in.shape[0], t_+1-Days_for_ewma))

    #calculate ewma for the first day t when data_in of Days_for_ewma is available
    ewma[:,0]=data_in[:,0]
    for id in range(1,Days_for_ewma):
        ewma[:,0]=data_in[:,id]+exp(coeff)@ewma

    ewma[:,0]=ewma[:,0]/sumcoeff

    #calculate ewma by using recurssion for the days afterward
    for i in range(1,(t_+1-Days_for_ewma)):
        ewma[:,i]=( data_in[:,i+Days_for_ewma-1]-exp(coeff@Days_for_ewma)@data_in[:,i-1] )/sumcoeff+exp((coeff))@ewma[:,i-1]
    return ewma


def assign_weight(signal, slope):
    #[weight]=assign_weight(signal, slope)
    #slope=1, long large signal
    #slope=-1, short large signal

    n_, l_=signal.shape

    #momentum strategy
    #weight is for allocating based on signal ranking among all N stocks each day
    weight = zeros((n_,l_))
    for i in range(l_):
        orig_pos=argsort(signal[:,i])
        weight[orig_pos,i]=slope*(-(n_+1)/2+arange(1,n_+1).T)*(2/(n_-1))
    return weight


def calc_units(weight, Prices, dailyExpVol):
    #[h_star]=calc_units(weight, Prices)

    cov_pnl=PnlVolatility_ewma(Prices)

    holding_by_weight=weight/Prices[:,-1]
    est_pnl_vol_daily=sqrt(holding_by_weight.T@cov_pnl@holding_by_weight)

    h_star= holding_by_weight@dailyExpVol/est_pnl_vol_daily
    return h_star, est_pnl_vol_daily
