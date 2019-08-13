from numpy import argsort, argmax, sign, arange, array, newaxis
from numpy.linalg import eig, svd

def pcacov(X):
    '''
    Python version of the MATLAB function which performs principal components analysis on the p-by-p covariance matrix X
     and returns the principal component coefficients,

    :param X:
    :return:
    '''
    w, v = eig(X)
    ind = argsort(-w)
    coeffs, latent = v[:, ind], w[ind]
    _,_,coeffs = svd(X)
    coeffs = coeffs.T
    p, d = coeffs.shape
    rowidx = array(argmax(abs(coeffs), axis=0))
    colidx = arange(0,d)
    colsign = sign(coeffs[rowidx,colidx])
    coeffs = coeffs*colsign[newaxis,...]

    return coeffs, latent