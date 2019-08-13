import numpy as np
from numpy import sum as npsum
from numpy import unique, zeros, where, abs


def TradeQuoteProcessing(t,dates_quotes,q_ask,p_ask,q_bid,p_bid,t_n,dates_trades,p_last,delta_q,delta_sgn,match,varargin=None):

    # This function processes the quote and the trade data as follows:
    # - modifies delta_sgn, adding the appropriate transaction sign where not
    # defined
    # - aggregates the match events, so that when multiple limit orders match
    # the same incoming market order, all the limit orders are rolled up in one
    # single event
    # - if there are repeated times, the function considers the last stored
    # observation for the quotes, while considers the cumulative volume, the
    # last price and the last sign for the trades
    # INPUTS
    # t             :[row vector] time vector for quotes
    # dates_quotes  :[row vector] date vector for quotes
    # q_ask         :[row vector] ask sizes
    # p_ask         :[row vector] best asks
    # q_bid         :[row vector] bid sizes
    # p_bid         :[row vector] best bids
    # t_n           :[row vector] time vector for trades
    # dates_trades  :[row vector] date vector for trades
    # p_last        :[row vector] last transaction prices
    # delta_q       :[row vector] flow of traded contracts' volumes
    # delta_sgn     :[row vector] trade sign flow
    # match         :[row vector] vector of match events
    # varargin{1}   :[row vector] number of separate limit orders on ask
    # varargin{2}   :[row vector] number of separate limit orders on bid
    # OUTPUTS
    # t_p             :[vector] (1 x k1_) processed time vector for quotes
    # dates_quotes_p  :[vector] (1 x k1_) processed date vector for quotes
    # q_ask_p         :[vector] (1 x k1_) processed ask sizes
    # p_ask_p         :[vector] (1 x k1_) processed ask prices
    # q_bid_p         :[vector] (1 x k1_) processed bid sizes
    # p_bid_p         :[vector] (1 x k1_) processed bid prices
    # t_n_p           :[vector] (1 x k2_) processed time vector for trades
    # dates_trades_p  :[vector] (1 x k2_) processed date vector for trades
    # p_last_p        :[vector] (1 x k2_) processed last transaction prices
    # delta_q_p       :[vector] (1 x k2_) processed flow of traded contracts' volumes
    # delta_sgn_p     :[vector] (1 x k2_) processed trade sign flow
    # varargout{1}    :[vector] (1 x k1_) processed number of separate limit orders on ask
    # varargout{2}    :[vector] (1 x k1_) processed number of separate limit orders on bid

    ## Code

    #QUOTES: if there are repeated times, consider the last stored observation
    #and delete the others
    t_unique=unique(t)
    k1_=len(t_unique)
    dates_quotes_p={}
    t_p=zeros((1,k1_))
    p_bid_p=zeros((1,k1_))
    p_ask_p=zeros((1,k1_))
    q_bid_p=zeros((1,k1_))
    q_ask_p=zeros((1,k1_))
    varargout = {1: zeros((1,k1_)), 2: zeros((1,k1_))}
    for k in range(k1_):
        index=where(t==t_unique[k])[0]
        dates_quotes_p[k]=dates_quotes[index[-1]]
        t_p[0,k]=t[index[-1]]
        p_bid_p[0,k]=p_bid[index[-1]]
        p_ask_p[0,k]=p_ask[index[-1]]
        q_bid_p[0,k]=q_bid[index[-1]]
        q_ask_p[0,k]=q_ask[index[-1]]
        if varargin is not None:
            varargout[2][0,k]=varargin[2][index[-1]]
            varargout[1][0,k]=varargin[1][index[-1]]

    #TRADES: set the sign of the transaction in delta_sgn where it is not defined:
    # -if traded price is closer to best ask the sign is "BUY", i.e. "+1"
    # -if traded price is closer to best bid the sign is "SELL", i.e. "-1"
    index=where(np.isnan(delta_sgn))[0]
    for i in range(len(index)):
        i_min=np.argmin(abs(t-t_n[index[i]]))
        if abs(p_last[index[i]]-p_ask[i_min])<abs(p_last[index[i]]-p_bid[i_min]):
            delta_sgn[index[i]]=+1
        else:
            delta_sgn[index[i]]=-1

    #TRADES: concatenate the "match" events
    index=where(~np.isnan(match))[0] #wheres the indices of elements NOT equal to NAN in vector match
    dates_trades_tmp=dates_trades[index]
    t_n_tmp=t_n[index]
    p_last_tmp=p_last[index]
    dv_tmp = zeros(len(index))
    dv_tmp[0]=npsum(delta_q[:index[0]+1])
    for k in range(1,len(index)):
        dv_tmp[k]=npsum(delta_q[index[k-1]+1:index[k]+1])
    dzeta_tmp = delta_sgn[index]

    #TRADES: if there are repeated times, consider the cumulative volume, the last price and the last sign
    t_n_unique=unique(t_n_tmp)
    k2_=len(t_n_unique)
    dates_trades_p={}
    t_n_p=zeros((1,k2_))
    p_last_p=zeros((1,k2_))
    delta_q_p=zeros((1,k2_))
    delta_sgn_p=zeros((1,k2_))
    for k in range(k2_):
        index=where(t_n_tmp==t_n_unique[k])[0]
        dates_trades_p[k]=dates_trades_tmp[index[-1]]
        t_n_p[0,k]=t_n_tmp[index[-1]]
        p_last_p[0,k]=p_last_tmp[index[-1]]
        delta_q_p[0,k]=npsum(dv_tmp[index])
        delta_sgn_p[0,k]=dzeta_tmp[index[-1]]
    return t_p, dates_quotes_p, q_ask_p, p_ask_p, q_bid_p, p_bid_p, t_n_p, dates_trades_p, p_last_p, delta_q_p, delta_sgn_p, varargout
