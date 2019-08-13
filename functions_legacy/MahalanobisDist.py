from numpy import eye, sqrt
from numpy.linalg import solve


def MahalanobisDist(x, mu, s2):
    # This function computes the Mahalanobis distance of a point x from a point mu
    # standardized by the symmetric and positive definite matrix s2
    # INPUTS
    # x        : [vector] (n_ x 1)  coordinates of a point
    # mu       : [vector] (n_ x 1)  coordinates of a point
    # s2       : [matrix] (n_ x n_) symmetric and positive definite matrix
    # OP
    # distance : [scalar] distance between the point x and the point mu,
    # standardized by the matrix s2

    # For details on the exercise, see here .

    invs2 = solve(s2,eye(s2.shape[0]))

    # distance = sqrt((x-mu).T@inv(s2)@(x-mu))
    distance = sqrt((x-mu).T@invs2@(x-mu))
    return distance
