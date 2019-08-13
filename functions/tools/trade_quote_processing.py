import numpy as np


def trade_quote_processing(t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_n,
                           dates_trades, p_last, delta_q, delta_sgn,
                           match, varargin=None):
    """For details, see here.

    Parameters
    ----------
        t : array, shape (k1_,)
        dates_quotes : array, shape (k1_,)
        q_ask : array, shape (k1_,)
        p_ask : array, shape (k1_,)
        q_bid : array, shape (k1_,)
        p_bid : array, shape (k1_,)
        t_n : array, shape (k3_,)
        dates_trades : array, shape (k3_,)
        p_last : array, shape (k3_,)
        delta_q : array, shape (k3_,)
        delta_sgn : array, shape (k3_,)
        match : array, shape (k3_,)
        varargin{1} : array, shape (k1_,)
        varargin{2} : array, shape (k1_,)

    Returns
    -------
        t_p : array, shape (k1_,)
        dates_quotes_p : array, shape (k1_,)
        q_ask_p : array, shape (k1_,)
        p_ask_p : array, shape (k1_,)
        q_bid_p : array, shape (k1_,)
        p_bid_p : array, shape (k1_,)
        t_n_p : array, shape (k2_,)
        dates_trades_p : array, shape (k2_,)
        p_last_p : array, shape (k2_,)
        delta_q_p : array, shape (k2_,)
        delta_sgn_p : array, shape (k2_,)
        varargout{1} : array, shape (k1_,)
        varargout{2} : array, shape (k1_,)

    """

    ## QUOTES: if there are repeated times, consider the last stored observation and delete the others

    t_unique = np.unique(t)
    k1_ = len(t_unique)
    dates_quotes_p = {}
    t_p = np.zeros(k1_)
    p_bid_p = np.zeros(k1_)
    p_ask_p = np.zeros(k1_)
    q_bid_p = np.zeros(k1_)
    q_ask_p = np.zeros(k1_)
    varargout = {1: np.zeros(k1_), 2: np.zeros(k1_)}
    for k in range(k1_):
        index = np.where(t == t_unique[k])[0]
        dates_quotes_p[k] = dates_quotes[index[-1]]
        t_p[k] = t[index[-1]]
        p_bid_p[k] = p_bid[index[-1]]
        p_ask_p[k] = p_ask[index[-1]]
        q_bid_p[k] = q_bid[index[-1]]
        q_ask_p[k] = q_ask[index[-1]]
        if varargin is not None:
            varargout[2][k] = varargin[2][index[-1]]
            varargout[1][k] = varargin[1][index[-1]]

    ## TRADES: set the sign of the transaction in delta_sgn where it is not defined: -if traded price is closer to best ask the sign is "BUY", i.e. "+1" -if traded price is closer to best bid the sign is "SELL", i.e. "-1"

    index = np.where(np.isnan(delta_sgn))[0]
    for i in range(len(index)):
        i_min = np.argmin(abs(t-t_n[index[i]]))
        if abs(p_last[index[i]]-p_ask[i_min]) < abs(p_last[index[i]] -\
                                                    p_bid[i_min]):
            delta_sgn[index[i]] = +1
        else:
            delta_sgn[index[i]] = -1

    ## TRADES: concatenate the "match" events wheres the indices of elements NOT equal to NAN in vector match

    index = np.where(~np.isnan(match))[0]
    dates_trades_tmp = dates_trades[index]
    t_n_tmp = t_n[index]
    p_last_tmp = p_last[index]
    dv_tmp = np.zeros(len(index))
    dv_tmp[0] = np.sum(delta_q[:index[0]+1])
    for k in range(1, len(index)):
        dv_tmp[k] = np.sum(delta_q[index[k-1]+1:index[k]+1])
    dzeta_tmp = delta_sgn[index]

    ## TRADES: if there are repeated times, consider the cumulative volume, the last price and the last sign

    t_n_unique = np.unique(t_n_tmp)
    k2_ = len(t_n_unique)
    dates_trades_p = {}
    t_n_p = np.zeros(k2_)
    p_last_p = np.zeros(k2_)
    delta_q_p = np.zeros(k2_)
    delta_sgn_p = np.zeros(k2_)
    for k in range(k2_):
        index = np.where(t_n_tmp == t_n_unique[k])[0]
        dates_trades_p[k] = dates_trades_tmp[index[-1]]
        t_n_p[k] = t_n_tmp[index[-1]]
        p_last_p[k] = p_last_tmp[index[-1]]
        delta_q_p[k] = np.sum(dv_tmp[index])
        delta_sgn_p[k] = dzeta_tmp[index[-1]]
    return t_p, dates_quotes_p, q_ask_p, p_ask_p, q_bid_p, p_bid_p, t_n_p, dates_trades_p, p_last_p, delta_q_p, delta_sgn_p, varargout
