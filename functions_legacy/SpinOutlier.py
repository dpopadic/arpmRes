import numpy as np
from numpy import arange, pi, array, ones, sort, cos, sin, argsort, sqrt, tile
from numpy.linalg import eig


def SpinOutlier(m,s2,scale,num):
    #Generate a number num of points along a circle centered in m
    #the radius is scale*sqrt(max(eigenvalues(s2))
    #INPUT
    # m :[vector] (n_ x 1)
    # s2 :[matrix) (n_ x n_)
    # scale: [scalar]
    # num: [scalar]
    #OP
    # out :[matrix] (n_ x num)
    ###########################################################

    n_=s2.shape[0]
    EigenValues, EigenVectors = eig(s2)
    EigenValues, idx = sort(EigenValues), argsort(EigenValues)
    EigenVectors = EigenVectors[:,idx]
    EigenValues=tile(EigenValues[n_-1,np.newaxis],(n_,1))
    Angle = arange(0,2*pi+2*pi/num,2*pi/num )
    Angle[-1] = 0
    circle = EigenVectors@np.diagflat(sqrt(EigenValues))@array([[cos(Angle[0])],[sin(Angle[0])]])
    for i in range(1,num):
        y = array([[cos(Angle[i])],[sin(Angle[i])]])
        circle = np.r_['-1',circle, EigenVectors@np.diagflat(sqrt(EigenValues))@y]
    out = m@ones((1, num)) + scale*circle
    return out
