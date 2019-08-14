# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_ellipsoid(m, s2, *, r=2, alpha=None, beta=None,
                   plot_axes=True, line_color='r', line_width=2,
                   plot_ellip=True, ellip_color=(.8, .8, .8), ellip_alpha = 0.5,
                   n_points=1000, point_color='b'):
    """For details, see here.

    Parameters
    ----------
        m : array, shape (3,)
        s2 : array, shape (3, 3)
        r : scalar, optional
        alpha : array, shape (n,)
        beta : array, shape (n,)
        plot_axes : boolean, optional
        line_color : char or tuple, optional
        line_width : scalar, optional
        plot_ellip : boolean, optional
        ellip_color : char or tuple, optional
        ellip_alpha : scalar, optional
        n_points : scalar, optional
        point_color : char or tuple, optional

    Returns
    -------
        f : matplotlib figure handle
        ax : matplotlib axis handle

    """

    lambda2, e = np.linalg.eigh(s2)
    s = e * np.sqrt(lambda2)

    plt.style.use('arpm')
    f, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'},
                         figsize=(14, 10))

    if n_points > 0:
        points = np.random.multivariate_normal(m, s2, n_points)
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
                '.', color=point_color)

    if plot_axes is True:
        x_axes_ = np.array([[0, r], [0, 0], [0, 0]])
        y_axes_ = np.array([[0, 0], [0, r], [0, 0]])
        z_axes_ = np.array([[0, 0], [0, 0], [0, r]])
        x_axes = s[0, 0] * x_axes_ + s[0, 1] * y_axes_ + s[0, 2] * z_axes_
        y_axes = s[1, 0] * x_axes_ + s[1, 1] * y_axes_ + s[1, 2] * z_axes_
        z_axes = s[2, 0] * x_axes_ + s[2, 1] * y_axes_ + s[2, 2] * z_axes_

        ax.plot(x_axes[0, :], y_axes[0, :], z_axes[0, :],
                color=line_color, lw=line_width)
        ax.plot(x_axes[1, :], y_axes[1, :], z_axes[1, :],
                color=line_color, lw=line_width)
        ax.plot(x_axes[2, :], y_axes[2, :], z_axes[2, :],
                color=line_color, lw=line_width)

    if plot_ellip is True:
        if alpha is None:
            alpha = np.linspace(0, 2*np.pi, 50)

        if beta is None:
            beta = np.linspace(0, np.pi, 50)

        # Cartesian coordinates that correspond to the spherical angles
        x_ball = r * np.outer(np.cos(alpha), np.sin(beta))
        y_ball = r * np.outer(np.sin(alpha), np.sin(beta))
        z_ball = r * np.outer(np.ones_like(alpha), np.cos(beta))
        x_ellip = s[0, 0] * x_ball + s[0, 1] * y_ball + s[0, 2] * z_ball
        y_ellip = s[1, 0] * x_ball + s[1, 1] * y_ball + s[1, 2] * z_ball
        z_ellip = s[2, 0] * x_ball + s[2, 1] * y_ball + s[2, 2] * z_ball

        ax.plot_surface(x_ellip, y_ellip, z_ellip, color=ellip_color, alpha=ellip_alpha)

    return f, ax

