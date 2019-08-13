from numpy import percentile, arange, zeros, array, interp
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def  ColorCodedFP(FP, Min_p=None, Max_p=None, GreyRange=None, Cmin=0, Cmax=1, ValueRange=[1, 0]):
    # This def creates a colormap and a color-specifying vector,
    # associating each probability value in FP with a certain grey gradation:
    # values with higher probability are associated with darker gradations of
    # grey values with lower probability are instead associated with lighter
    # gradations.
    #  INPUTS
    #  FP                   : [vector](1 x t_end) vector of flexible probabilities
    #  Min_p      (optional): [scalar] lower threshold: to probabilities <=Min_p is associated the lightest grey
    #  Max_p      (optional): [scalar] upper threshold: to probabilities >=Max_p is associated the darkest grey
    #  GreyRange  (optional): [column vector] it defines the hues of grey in the Colormap
    #                       - first entry: darkest gray (default: 0 = black)
    #                       - last entry: lightest gray (default: 0.8)
    #                       - step: colormap step (default 0.01 if more hues are needed set it to a smaller value)
    #  Cmin       (optional): [scalar] value associated to the darkest grey (default 0)
    #  Cmax       (optional): [scalar] value associated to the lightest grey (default 20)
    #  ValueRange (optional): [vector](1 x 2) range of values associated to hues of grey in the middle (default [20 0])
    #  OUTPUTS
    #  Colormap             : [matrix] this is the colormap to set before plotting the scatter
    #  FPColors             : [vector](t_end x 1) contains the colors to set as input argoment of the def "scatter"
    #                         the values in FPColors are linearly mapped to the colors in Colormap.

    ## Code

    if Min_p is None:
        Min_p = percentile(FP.T, 1)
    if Max_p is None:
        Max_p = percentile(FP.T, 99)
    if GreyRange is None:
        GreyRange = arange(0,0.81,0.01)

    GreyRange = GreyRange.ravel()
    Colormap = truncate_colormap(plt.get_cmap('gray'),min(GreyRange),max(GreyRange))

    #scatter colors
    C = zeros((FP.shape[1], 1))
    xvals = array([Min_p, Max_p])
    yvals = array(ValueRange)
    for t in range(FP.shape[1]):
        if FP[:, t] >= Max_p:
            C[t] = Cmin
        elif FP[:, t] <= Min_p:
             C[t] = Cmax
        else:
            C[t] = interp(FP[0, t], xvals, yvals)

    FPColors = np.squeeze(C)

    return Colormap, FPColors

