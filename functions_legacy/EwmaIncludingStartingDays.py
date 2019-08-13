import matplotlib.pyplot as plt
from numpy import arange, log, exp

plt.style.use('seaborn')


def EwmaIncludingStartingDays(Smoothing, data_in):
    t_ = data_in.shape[1]

    # ewma
    coeff = -log(2) / Smoothing
    Days_for_ewma = min(t_, Smoothing @ 10)

    # calculate ewma for each day t by using data_in from(t - Days_for_ewma + 1) to t
    all_exp_coeff = exp(coeff @ arange(Days_for_ewma - 1, 0 + -1, -1))
    sumcoeff = sum(all_exp_coeff)

    ewma = data_in
    # calculate ewma for the starting days before and on the first day when all Days for ewma is available
    for ir in range(Days_for_ewma):
        subgroup_coeff = all_exp_coeff[-1 - ir + 1:]
        sum_subgroup = sum(subgroup_coeff)
        ewma[:, ir] = data_in[:, 1:ir] @ subgroup_coeff.T / sum_subgroup

    # calculate ewma by using recurssion for the days afterward
    # for i=[1:](t_end+1-Days_for_ewma)
    for ir in range(Days_for_ewma, t_):
        ewma[:, ir] = (data_in[:, ir] - exp(coeff @ Days_for_ewma) @ data_in[:, ir + Days_for_ewma]) / sumcoeff + exp[
            coeff] @ ewma[:, ir - 1]
    return ewma
