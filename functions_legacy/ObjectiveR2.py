from numpy import trace, array, diag, sqrt, atleast_2d, ix_, diagflat
from numpy.linalg import solve


def ObjectiveR2(data,v_k):
    # Compute the r-square from the covariance matrix of the recovered market
    # variables X and the k selected factors Z(v_k)
    #  INPUTS
    # data  :[struct] struct with two fields:
    #   1) data.covXZ :[matrix]((n_+m_) x (n_+m_)) covariance matrix of the joint distribution of random vector (XZ)
    #   2) data.n_    :[scalar] dimension of random vector X
    # v_k   :[vector] indeces of selected factors
    #  OPS
    # R2   :[scalar] r-square value

    # For details on the exercise, see here .

    ## Code
    covXZ = data.covXZ
    n_ = data.n_

    if isinstance(v_k,int) or isinstance(v_k,float):
        v_k = array([v_k])

    # extract covariance matrices from covXZ
    covX = covXZ[:n_,:n_]
    covXZ_k = atleast_2d(covXZ[:n_,v_k+n_])
    covZ_k = atleast_2d(covXZ[ix_(v_k+n_,v_k+n_)])

    # compute correlation matrices
    diagX = diagflat(1/sqrt(diag(covX)))
    diagZ_k = diagflat(1/sqrt(diag(covZ_k)))

    crXZ_k = diagX@covXZ_k@diagZ_k
    crZ_k = diagZ_k@covZ_k@diagZ_k

    # compute r-square
    R2 = trace(crXZ_k@ solve(crZ_k,crXZ_k.T))/n_
    return R2
