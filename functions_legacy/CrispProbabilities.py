from numpy import max as npmax, min as npmin, sum as npsum
from numpy import interp, unique, sort, zeros, tile, array

from functions_legacy.HFPcdf import HFPcdf
from functions_legacy.ARPM_utils import matlab_percentile


def CrispProbabilities(Conditioner):
    # Conditioning via crisp Flexible Probabilities
    # INPUT
    # Conditioner  :[struct] with fields
    #  Series      :[vector] (1 x t_end) time series of the conditioner
    #  TargetValue :[vector] (1 x k_) target values for the conditioner
    #  Leeway:     :[scalar] (alpha) probability contained in the range, which is symmetric around the target value.
    # OUTPUT
    # crisp_prob   :[matrix] (k_ x t_end) crisp probabilities for each of the k_ target values
    # z_lb         :[vector] (k_ x 1) range lower bound for each of the k_ target values
    # z_ub         :[vector] (k_ x 1) range upper bound for each of the k_ target values

    # For details on the exercise, see here .
    ######################################################################################

    Z = Conditioner.Series
    zz = Conditioner.TargetValue
    if isinstance(zz, float) or isinstance(zz, int):
        zz = array([[zz]])
    alpha = Conditioner.Leeway

    t_ = Z.shape[1]
    zz_len = zz.shape[1]

    z = unique(sort(Z).T).reshape((1, -1))
    ecdf_z = unique(HFPcdf(z, Z, tile(1 / t_, (1, t_))))
    cdf_zz = interp(zz.flatten(), z.flatten(), ecdf_z.flatten())

    # upper and lower quantiles of z
    zmin = matlab_percentile(z.flatten(), 100 * alpha / 2)
    zmax = matlab_percentile(z.flatten(), 100 * (1 - (alpha / 2)))

    z_lb = zeros((zz_len, 1))
    z_ub = zeros((zz_len, 1))
    p = zeros((zz_len, t_))
    pp = zeros((zz_len, t_))

    for i in range(zz_len):
        cdf_zz[cdf_zz >= 1 - alpha / 2] = 1 - alpha / 2
        cdf_zz[cdf_zz <= alpha / 2] = alpha / 2

        z = zz[0, i]
        if z <= zmin:
            z_lb[i] = npmin(Z)
            z_ub[i] = matlab_percentile(Z.flatten(), 100 * (cdf_zz[i] + (alpha / 2)))
        elif z >= zmax:
            z_lb[i] = matlab_percentile(Z.flatten(), 100 * (cdf_zz[i] - (alpha / 2)))
            z_ub[i] = npmax(Z)
        else:
            z_lb[i] = matlab_percentile(Z.flatten(), 100 * (cdf_zz[i] - (alpha / 2)))
            z_ub[i] = matlab_percentile(Z.flatten(), 100 * (cdf_zz[i] + (alpha / 2)))
        # crisp probabilities
        for t in range(t_):
            if Z[0, t] <= z_ub[i] and Z[0, t] >= z_lb[i]:
                pp[i, t] = 1
            else:
                pp[i, t] = 0
        p[i, :] = pp[i, :] / npsum(pp[i, :])

    crisp_prob = p
    return crisp_prob, z_lb, z_ub
