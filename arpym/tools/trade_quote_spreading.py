import numpy as np


def trade_quote_spreading(t_ms, t, q_ask, p_ask, q_bid, p_bid, t_n, p_last, q,
                          sgn, varargin=None):
    """For details, see here.

    Parameters
    ----------
        t_ms : array, shape (k_,)
        t : array, shape (t_,)
        q_ask : array, shape (t_,)
        p_ask : array, shape (t_,)
        q_bid : array, shape (t_,)
        p_bid : array, shape (t_,)
        t_n : array, shape (k1_,)
        p_last : array, shape (k1_,)
        q : array, shape (k1_,)
        sgn : array, shape (k1_,)
        varargin{1} : array, shape (t_,)
        varargin{2} : array, shape (t_,)

    Returns
    -------
        q_ask_s : array, shape (k_,)
        p_ask_s : array, shape (k_,)
        q_bid_s : array, shape (k_,)
        p_bid_s : array, shape (k_,)
        p_last_s : array, shape (k_,)
        q_s : array, shape (k_,)
        sgn_s : array, shape (k_,)
        varargout{1} : array, shape (k_,)
        varargout{2} : array, shape (k_,)

    """

    k_ = len(t_ms)

    ## Step 1: define new variables

    q_ask_s = np.zeros(k_)
    p_ask_s = np.zeros(k_)
    q_bid_s = np.zeros(k_)
    p_bid_s = np.zeros(k_)
    p_last_s = np.zeros(k_)
    p_last_s[:] = np.NaN
    q_s = np.zeros(k_)
    sgn_s = np.zeros(k_)
    varargout = {1: np.zeros(k_), 2: np.zeros(k_)}

    ## Step 2: initialize variables

    q_ask_s[0] = q_ask[0]
    p_ask_s[0] = p_ask[0]
    q_bid_s[0] = q_bid[0]
    p_bid_s[0] = p_bid[0]
    if varargin is not None:
        varargout[2] = varargin[2]
        varargout[1] = varargin[1]

    i_t_n = np.where(abs(t_n-t_ms[0]) < 1.0e-9)[0]
    if (len(i_t_n) == 1):
        p_last_s[0] = p_last[i_t_n]
        q_s[0] = q[i_t_n]
        sgn_s[0] = sgn[i_t_n]
    elif (len(i_t_n) > 1):
        i_t_n1 = np.where(abs(t_n-t_ms[0]) < 1.0e-10)[0]
        p_last_s[0] = p_last[i_t_n1]
        q_s[0] = q[i_t_n1]
        sgn_s[0] = sgn[i_t_n1]

    ## Step 3: update variables

    for k in range(1, k_):
        i_t = np.where(abs(t-t_ms[k]) < 1.0e-9)[0]
        if i_t.size == 0:
            if varargin is not None:
                varargout[2][k] = varargout[2][k-1]
                varargout[1][k] = varargout[1][k-1]
            q_ask_s[k] = q_ask_s[k-1]
            p_ask_s[k] = p_ask_s[k-1]
            q_bid_s[k] = q_bid_s[k-1]
            p_bid_s[k] = p_bid_s[k-1]
        elif len(i_t) == 1:
            if varargin is not None:
                varargout[2][k] = varargin[2][i_t]
                varargout[1][k] = varargin[1][i_t]
            q_ask_s[k] = q_ask[i_t]
            p_ask_s[k] = p_ask[i_t]
            q_bid_s[k] = q_bid[i_t]
            p_bid_s[k] = p_bid[i_t]
        else:
            i_t1 = np.where(abs((t-t_ms[k])) < 1.0e-10)[0]
            if varargin is not None:
                varargout[2][k] = varargin[2][i_t1]
                varargout[1][k] = varargin[1][i_t1]
            q_ask_s[k] = q_ask[i_t1]
            p_ask_s[k] = p_ask[i_t1]
            q_bid_s[k] = q_bid[i_t1]
            p_bid_s[k] = p_bid[i_t1]
        i_t_n = np.where(abs(t_n-t_ms[k]) < 1.0e-6)[0]
        if len(i_t_n) == 1:
            p_last_s[k] = p_last[i_t_n]
            q_s[k] = q[i_t_n]
            sgn_s[k] = sgn[i_t_n]
        elif len(i_t_n) > 1:
            i_t_n1 = np.where(abs(t_n-t_ms[k]) < 1.0e-10)[0]
            p_last_s[k] = p_last[i_t_n1]
            q_s[k] = q[i_t_n1]
            sgn_s[k] = sgn[i_t_n1]
    return q_ask_s, p_ask_s, q_bid_s, p_bid_s, p_last_s, q_s, sgn_s, varargout
