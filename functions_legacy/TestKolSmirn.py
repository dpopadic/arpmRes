from numpy import unique, sort, percentile, sqrt, tile, linspace

from scipy.interpolate import interp1d

from HFPcdf import HFPcdf
from RandomSplit import RandomSplit


def TestKolSmirn(y):
    # This function returns the Kolmogorov-Smirnov test for a given time series
    # INPUTS
    #  y  :[vector](1 x t_end) time series to be analyzed
    # OUTPUTS
    #  y1        :[vector](1 x ~t_end/2) first partition of vector y
    #  y2        :[vector](1 x ~t_end/2) second partition of vector y
    #  band_int    :[row vector] x-axis values for the band
    #  cdf_1     :[vector](1 x ~t_end/2) empirical cdf of vector y1
    #  cdf_2     :[vector](1 x ~t_end/2) empirical cdf of vector y2
    #  up_band   :[row vector] y-axis values of the upper band
    #  low_band  :[row vector] y-axis values of the lower band

    ## Code
    # split vector y into two mutually exclusive partitions, of about the same length
    y1, y2 = RandomSplit(y)
    k1_ = y1.shape[1]
    k2_ = y2.shape[1]

    # compute their empirical cumulative distribution functions
    x1 = unique(sort(y1)).reshape(1, -1)
    f1 = unique(HFPcdf(x1, y1, tile([1 / k1_], (1, k1_))).T).reshape(1, -1)
    x2 = unique(sort(y2)).reshape(1, -1)
    f2 = unique(HFPcdf(x2, y2, tile(1 / k2_, (1, k2_))).T).reshape(1, -1)

    # set the interval values for the Kolmogorov-Smirnov (upper and lower) band
    x_lim1 = percentile(y.T, 1.5)
    x_lim2 = percentile(y.T, 98.5)
    band_int = linspace(x_lim1, x_lim2, 10001)  # x-axis values for the band

    interp1 = interp1d(x1[0], f1[0], fill_value='extrapolate')
    cdf_band_1 = interp1(band_int)
    cdf_1 = interp1(y1)

    interp2 = interp1d(x2[0], f2[0], fill_value='extrapolate')
    cdf_band_2 = interp2(band_int)
    cdf_2 = interp2(y2)

    # build the band for Kolmogorov-Smirnov test
    band_mid = 0.5 * (cdf_band_1 + cdf_band_2)
    up_band = band_mid + 0.5 * (1.22 / sqrt(k1_))
    low_band = band_mid - 0.5 * (1.22 / sqrt(k1_))
    return y1, y2, band_int, cdf_1, cdf_2, up_band, low_band
