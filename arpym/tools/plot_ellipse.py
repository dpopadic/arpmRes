# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from arpym.tools.pca_cov import pca_cov


def plot_ellipse(m, s2, *, r=1, n_=1000, display_ellipse=True, plot_axes=False,
                 plot_tang_box=False, color='k', line_width=2):
    """For details, see here.

    Parameters
    ----------
        mu : array, shape (2,)
        sigma2 : array, shape (2,2)
        r : scalar, optional
        n_ : scalar, optional
        display_ellipse : boolean, optional
        plot_axes : boolean, optional
        plot_tang_box : boolean, optional
        color : char, optional
        line_width : scalar, optional

    Returns
    -------
        x : array, shape (n_points,2)

    """

    # Step 1: compute the circle with the radius r

    theta = np.arange(0, 2 * np.pi, (2*np.pi)/n_)
    y = [r * np.cos(theta), r * np.sin(theta)]

    # Step 2: spectral decomposition of s2

    e, lambda2 = pca_cov(s2)
    Diag_lambda = np.diagflat(np.sqrt(np.maximum(lambda2, 0)))

    # Step 3: compute the ellipse as affine transformation of the circle

    # stretch
    x = Diag_lambda@y
    # rotate
    x = e@x
    # translate
    x = m.reshape((2, 1)) + x

    if display_ellipse:
        plt.plot(x[0], x[1], lw=line_width, color=color)
        plt.grid(True)

        # Step 4: plot the tangent box
        if plot_tang_box:
            sigvec = np.sqrt(np.diag(s2))

            rect = patches.Rectangle(m-r * sigvec, 2*r * sigvec[0],
                                     2*r*sigvec[1], fill=False,
                                     linewidth=line_width)
            ax = plt.gca()
            ax.add_patch(rect)

        # Step 5: plot the principal axes
        if plot_axes:
            # principal axes
            plt.plot([m[0]-r*np.sqrt(lambda2[0])*e[0, 0],
                      m[0]+r*np.sqrt(lambda2[0])*e[0, 0]],
                     [m[1]-r*np.sqrt(lambda2[0])*e[1, 0],
                      m[1]+r*np.sqrt(lambda2[0])*e[1, 0]], color=color,
                     lw=line_width)

            plt.plot([m[0]-r*np.sqrt(lambda2[1])*e[0, 1],
                      m[0]+r*np.sqrt(lambda2[1])*e[0, 1]],
                     [m[1]-r*np.sqrt(lambda2[1])*e[1, 1],
                      m[1]+r*np.sqrt(lambda2[1])*e[1, 1]], color=color,
                     lw=line_width)
            plt.axis('equal')

    return x.T
