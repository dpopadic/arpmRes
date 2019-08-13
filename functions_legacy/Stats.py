from numpy import ones, zeros, cumsum, argsort, sort, sqrt, where
from numpy import sum as npsum

from FPmeancov import FPmeancov


def Stats(epsi, FP=None):
    # Given a time series (epsi) and the associated probabilities FP,
    # this def computes the statistics: mean,
    # standard deviation,VaR,CVaR,skewness and kurtosis.
    # INPUT
    # epsi :[vector] (1 x t_end)
    # FP   :[matrix] (q_ x t_end) statistics are computed for each of the q_ sets of probabilities.
    # OUTPUT
    # m     :[vector] (q_ x 1) mean of epsi with FP (for each set of FP)
    # stdev :[vector] (q_ x 1) standard deviation of epsi with FP (for each set of FP)
    # VaR   :[vector] (q_ x 1) value at risk with FP
    # CVaR  :[vector] (q_ x 1) conditional value at risk with FP
    # sk    :[vector] (q_ x 1) skewness with FP
    # kurt  :[vector] (q_ x 1) kurtosis with FP
    ###########################################################################

    # size check
    if epsi.shape[0] > epsi.shape[1]:
        epsi = epsi.T  # eps: row vector
    if FP.shape[1] != epsi.shape[1]:
        FP = FP.T

    # if FP argument is missing, set equally weighted FP
    t_ = epsi.shape[1]
    if FP is None:
        FP = ones((1, t_)) / t_

    q_ = FP.shape[0]

    m = zeros((q_, 1))
    stdev = zeros((q_, 1))
    VaR = zeros((q_, 1))
    CVaR = zeros((q_, 1))
    sk = zeros((q_, 1))
    kurt = zeros((q_, 1))

    for q in range(q_):
        m[q] = (epsi * FP[[q], :]).sum()
        stdev[q] = sqrt(npsum(((epsi - m[q]) ** 2) * FP[q, :]))
        SortedEps, idx = sort(epsi), argsort(epsi)
        SortedP = FP[[q], idx]
        VarPos = where(cumsum(SortedP) >= 0.01)[0][0]
        VaR[q] = -SortedEps[:, VarPos]
        CVaR[q] = -FPmeancov(SortedEps[[0], :VarPos + 1], SortedP[:, :VarPos + 1].T / npsum(SortedP[:, :VarPos + 1]))[0]
        sk[q] = npsum(FP[q, :] * ((epsi - m[q]) ** 3)) / (stdev[q] ** 3)
        kurt[q] = npsum(FP[q, :] * ((epsi - m[q]) ** 4)) / (stdev[q] ** 4)

    return m, stdev, VaR, CVaR, sk, kurt
