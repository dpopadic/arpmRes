from matplotlib.pyplot import plot, grid, axis
from numpy import cos, sin, pi, linspace, diag, sqrt, tile, array, sort, argsort, diagflat, maximum
from numpy.linalg import eig

import numpy as np
np.seterr(invalid='ignore')


def PlotTwoDimEllipsoid(mu, sigma2, r=1, PlotAxes=False, PlotTangBox=False, color='k', linewidth=2, PlotEll=True,
                        n_points=1000, fig=None, ax=None):
    '''This def creates and plots the two dimensional ellipsoid:
    (x - mu).T * (sigma2**(-1)) * (x - mu) = r**2
     INPUTS
      mu          : [vector] (2 x 1)
      sigma2      : [matrix] (2 x 2) symmetric and positive definite matrix
      r           : [scalar] radius of the ellipsoid
      PlotAxes    : [boolean] if true then the principal axes are plotted
      PlotTangBox : [boolean] if true then the tangent box is plotted.
      color       : [char] color of the line defining the ellipsoid
      linewidth   : [scalar] width of the line defining the ellipsoid
      PlotEll     : [boolean] if true then the ellipsoid is plotted
      n_points    : [scalar] number of points of the ellipsoid
     OUTPUTS
      ell_handle  : [figure handle] ellipsoid
      ell_points  : [matrix] (2 x n_points) points of the ellipsoid
      ax1, ax2    : [figure handle] principal axes
'''

    # For details on the exercise, see here .
    ## Code
    theta = linspace(0, 2 * pi, n_points)

    # compute the initial sphere
    y = [r * cos(theta), r * sin(theta)]

    # principal axes
    y_axes1 = array([[-r, r], [0, 0]])
    y_axes2 = array([[0, 0], [-r, r]])

    # spectral decomposition of sigma2
    Diag_lambda2, e = eig(sigma2)
    lambda2, order = sort(Diag_lambda2), argsort(Diag_lambda2)
    e = e[:, order]
    Diag_lambda = diagflat(sqrt(maximum(lambda2,0)))

    # compute the ellipsoid as affine transformation of the sphere
    u = e@Diag_lambda@y
    u_axes1 = e@Diag_lambda@y_axes1
    u_axes2 = e@Diag_lambda@y_axes2
    ell_points = tile(mu, (1, n_points)) + u

    # if fig is None and ax is None:
    #     fig = figure()
    #     ax = fig.add_subplot()
    # elif ax is None:
    #     ax = gca()
    # plot the ellipsoid
    if PlotEll:
        ell_handle = plot(ell_points[0], ell_points[1], lw=linewidth, color=color)
        grid(True)
    else:
        ell_handle = None

    # plot the tangent box
    if PlotTangBox:
        sigvec = sqrt(diag(sigma2))

        tangBox_low = [[mu[0] - r * sigvec[0],  mu[0] + r * sigvec[0]], [mu[1] - r * sigvec[1], mu[1] - r * sigvec[1]]]
        tangBox_up = [[mu[0] - r * sigvec[0],  mu[0] + r * sigvec[0]], [mu[1] + r * sigvec[1], mu[1] + r * sigvec[1]]]
        tangBox_left = [[mu[0] - r * sigvec[0], mu[0] - r * sigvec[0]], [mu[1] - r * sigvec[1],  mu[1] + r * sigvec[1]]]
        tangBox_right = [[mu[0] + r * sigvec[0], mu[0] + r * sigvec[0]], [mu[1] - r * sigvec[1],  mu[1] + r * sigvec[1]]]

        h1 = plot(tangBox_low[0], tangBox_low[1], color=color, lw=linewidth)
        h2 = plot(tangBox_up[0], tangBox_up[1], color=color, lw=linewidth)
        h3 = plot(tangBox_left[0], tangBox_left[1], color=color, lw=linewidth)
        h4 = plot(tangBox_right[0], tangBox_right[1], color=color, lw=linewidth)

    # plot the principal axes
    if PlotAxes:
        ax1 = plot(u_axes1[0] + mu[0], u_axes1[1] + mu[1], color=color, lw=linewidth)
        ax2 = plot(u_axes2[0] + mu[0], u_axes2[1] + mu[1], color=color, lw=linewidth)
        axis('equal')
    else:
        ax1 = None
        ax2 = None

    return ell_handle, ell_points, ax1, ax2
