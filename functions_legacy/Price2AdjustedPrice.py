from numpy import where, diff, log, copy


def Price2AdjustedPrice(date, value_close, div=None, fwd=1):
    # This function returns the dividend adjusted values of a given stocks series
    # INPUTS
    #  date         :[vector](1 x t_end) dates (of the stock's closing values)
    #  value_close  :[vector](1 x t_end) stock's closing values
    #  div          :[matrix](2 x t_end) -First row stores dates on which dividends are issued
    #                                 -Second row stores dividends value
    #  fwd          :[integer](1 x 1) if fwd=1 (default) the value is forward adjusted, otherwise it is backward adjusted
    #
    # OUTPUTS
    #  value_adj  :[vector](1 x t_end) dividend adjusted values
    #  ret_adj    :[vector](1 x t_end-1) log-returns computed from dividend adjusted values

    # For details on the exercise, see here .

    if div is not None and div.ndim == 1:
        div = div.reshape(-1,1)
    if div is not None and div.size!=0:
        div_date = div[[0], :]
        div_value = div[[1], :]
        value_adj = copy(value_close)
        if fwd == 1:
            for k in range(max(div_value.shape)):
                t_k = where(date[0]<div_date[0,k])[0]
                if len(t_k) > 0:
                    t_k = t_k[-1]
                    if date[0,t_k]:
                        value_adj[0,t_k+1:] = value_adj[0,t_k+1:] / (1 - div_value[0,k] / value_close[0, t_k])
        else:
            for k in range(max(div_value.shape)):
                t_k = where(date[0]<div_date[0,k])[0][-1]
                if date[0,t_k]:
                    value_adj[0,:t_k] = value_adj[0,:t_k] * (1 - div_value[0,k] / value_close[0, t_k])

    else:
        value_adj = copy(value_close)
    ret_adj=diff(log(value_adj)) # adjusted log-returns
    return value_adj, ret_adj
