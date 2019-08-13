# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def colormap_fp(p, min_p=None, max_p=None, grey_range=None, c_min=0, c_max=1,
                value_range=[1, 0]):
    """For details, see here.

    Parameters
    ----------
        p : array, shape (t_,)
        min_p : scalar, optional
        max_p : scalar, optional
        grey_range : array, shape (3, ), optional
        c_min : scalar, optional
        c_max : scalar, optional
        value_range : array, shape (2,), optional

    Returns
    -------
        color_map : array
        fp_colors : array, shape (t_,)

    """

    if min_p is None:
        min_p = np.percentile(p.T, 1)
    if max_p is None:
        max_p = np.percentile(p.T, 99)
    if grey_range is None:
        grey_range = np.arange(0, 0.81, 0.01)

    grey_range = grey_range.ravel()

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    color_map = truncate_colormap(plt.get_cmap('gray'), min(grey_range),
                                  max(grey_range))

    # scatter colors
    fp_colors = np.zeros(p.shape[0])
    xvals = np.array([min_p, max_p])
    yvals = np.array(value_range)
    for t in range(p.shape[0]):
        if p[t] >= max_p:
            fp_colors[t] = c_min
        elif p[t] <= min_p:
            fp_colors[t] = c_max
        else:
            fp_colors[t] = np.interp(p[t], xvals, yvals)

    return color_map, fp_colors
