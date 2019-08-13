import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from numpy import zeros

plt.style.use('seaborn')


def PlotTwoDimBand(mu, sigma_u, u, r=1, color='k', linewidth=2, PlotBand=True):

    # This function creates and plots the two dimensional uncertainty band:
    #  U_{X}[r] = {x = mu_{X} + r@sigma_{u.T@X}@u, |u| = 1}
    #  INPUTS
    #   mu          : [vector] (2 x 1) vector of locations of X_1 and X_2
    #   sigma_u     : [vector] (n_points x 1) vector of dispersions along the projections of u.T@X
    #   u           : [matrix] (2 x n_points) unit-length vectors
    #   r           : [scalar] scale of the band
    #   color       : [char] color of the line defining the band
    #   linewidth   : [scalar] width of the line defining the band
    #   PlotBand    : [boolean] if true then the band is plotted
    #  OPS
    #   band_handle : [figure handle]
    #   band_points : [matrix] (2 x n_points) points of the band
    # NOTE: Set the matrix of directions u as follows
    # - theta = linspace(0,2 * pi, n_points)
    # - u = [cos((theta)) sin((theta))]

    # For details on the exercise, see here .

    ## Code

    # Compute the points of the band
    n_points = u.shape[1]
    band_points = zeros((2, n_points))

    for n in range(n_points):
        band_points[:, [n]] =  mu + r*sigma_u[n]*u[:, [n]]

    # Plot the band
    if PlotBand:
        band_handle = plot(band_points[0, :], band_points[1, :], color=color, lw=linewidth)
        plt.grid(True)
    else:
        band_handle = None
    return band_handle, band_points