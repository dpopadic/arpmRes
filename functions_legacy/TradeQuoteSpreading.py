import numpy as np
from numpy import zeros, where, abs


def TradeQuoteSpreading(t_ms,t,q_ask,p_ask,q_bid,p_bid,t_n,p_last,q,sgn,varargin=None):
    # This function spreads the quote and the trade events, contained in the
    # input arguments, on the wall-clock-time vector t_ms.
    # More specifically, it defines new quote variables that correspond to the
    # old quotes if t_ms=t, and are equal to the quotes at the previous time
    # for the other values of t_ms then defines new trade variables that
    # correspond to the old trades if t_ms=t_n, and are zeros for the other
    # values of t_ms.
    # INPUTS
    # t_ms         :[vector] (1 x k_) wall-clock-time vector expressed in milliseconds
    # t            :[row vector] time vector for quotes
    # q_ask        :[row vector] ask sizes
    # p_ask        :[row vector] best asks
    # q_bid        :[row vector] bid sizes
    # p_bid        :[row vector] best bids
    # t_n          :[row vector] time vector for trades
    # p_last       :[row vector] last transaction price
    # q            :[row vector] cumulative volume of traded contracts
    # sgn          :[row vector] cumulative trade sign
    # varargin{1}  :[row vector] number of separate limit orders on ask
    # varargin{2}  :[row vector] number of separate limit orders on bid
    # OUTPUTS
    # q_ask_s        :[vector] (1 x k_) ask sizes, spreaded on t_ms
    # p_ask_s        :[vector] (1 x k_) best asks, spreaded on t_ms
    # q_bid_s        :[vector] (1 x k_) bid sizes, spreaded on t_ms
    # p_bid_s        :[vector] (1 x k_) best bids, spreaded on t_ms
    # p_last_s       :[vector] (1 x k_) last transaction prices, spreaded on t_ms
    # q_s            :[vector] (1 x k_) cumulative volume of traded contracts, spreaded on t_ms
    # sgn_s          :[vector] (1 x k_) cumulative trade sign, spreaded on t_ms
    # varargout{1}   :[vector] (1 x k_) number of separate limit orders on ask, spreaded on t_ms
    # varargout{2}   :[vector] (1 x k_) number of separate limit orders on bid, spreaded on t_ms

    ## Code

    k_=len(t_ms)
    #define new variables
    q_ask_s=zeros((1,k_))
    p_ask_s=zeros((1,k_))
    q_bid_s=zeros((1,k_))
    p_bid_s=zeros((1,k_))
    p_last_s=zeros((1,k_))
    p_last_s[:] = np.NaN
    q_s=zeros((1,k_))
    sgn_s=zeros((1,k_))
    varargout = {1: zeros((1,k_)), 2: zeros((1,k_))}

    #initialize variables
    q_ask_s[0,0]=q_ask[0]
    p_ask_s[0,0]=p_ask[0]
    q_bid_s[0,0]=q_bid[0]
    p_bid_s[0,0]=p_bid[0]
    if varargin is not None:
        varargout[2][0]= varargin[2][0]
        varargout[1][0] = varargin[1][0]

    i_t_n=where(abs(t_n-t_ms[0])<1.0e-9)[0]
    if (len(i_t_n)==1):
        p_last_s[0,0]=p_last[i_t_n]
        q_s[0,0]=q[i_t_n]
        sgn_s[0,0]=sgn[i_t_n]
    elif (len(i_t_n)>1):
        i_t_n1=where(abs(t_n-t_ms[0])<1.0e-10)[0]
        p_last_s[0,0]=p_last[i_t_n1]
        q_s[0,0]=q[i_t_n1]
        sgn_s[0,0]=sgn[i_t_n1]

    #update variables
    for k in range(1, k_):
        i_t=where(abs(t-t_ms[k])<1.0e-9)[0]
        if i_t.size == 0:
            if varargin is not None:
                varargout[2][0,k] = varargout[2][0,k-1]
                varargout[1][0,k] = varargout[1][0,k-1]
            q_ask_s[0,k]=q_ask_s[0,k-1]
            p_ask_s[0,k]=p_ask_s[0,k-1]
            q_bid_s[0,k]=q_bid_s[0,k-1]
            p_bid_s[0,k]=p_bid_s[0,k-1]
        elif len(i_t) == 1:
            if varargin is not None:
                varargout[2][0,k] = varargin[2][i_t]
                varargout[1][0,k] = varargin[1][i_t]
            q_ask_s[0,k]=q_ask[i_t]
            p_ask_s[0,k]=p_ask[i_t]
            q_bid_s[0,k]=q_bid[i_t]
            p_bid_s[0,k]=p_bid[i_t]
        else:
            i_t1=where(abs((t-t_ms[k]))<1.0e-10)[0]
            if varargin is not None:
                varargout[2][0,k] = varargin[2][i_t1]
                varargout[1][0,k] = varargin[1][i_t1]
            q_ask_s[0,k]=q_ask[i_t1]
            p_ask_s[0,k]=p_ask[i_t1]
            q_bid_s[0,k]=q_bid[i_t1]
            p_bid_s[0,k]=p_bid[i_t1]
        i_t_n=where(abs(t_n-t_ms[k])<1.0e-9)[0]
        if len(i_t_n)==1:
            p_last_s[0,k]=p_last[i_t_n]
            q_s[0,k]=q[i_t_n]
            sgn_s[0,k]=sgn[i_t_n]
        elif len(i_t_n)>1:
            i_t_n1=where(abs(t_n-t_ms[k])<1.0e-10)[0]
            p_last_s[0,k]=p_last[i_t_n1]
            q_s[0,k]=q[i_t_n1]
            sgn_s[0,k]=sgn[i_t_n1]
    return q_ask_s,p_ask_s,q_bid_s,p_bid_s,p_last_s,q_s,sgn_s,varargout
