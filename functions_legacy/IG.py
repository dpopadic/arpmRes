from numpy import where, sqrt
from numpy.random import rand, randn


def IG(l,m,j_):

    n=randn(j_,1)
    y=n**2
    x = m + (.5*m*m/l)*y - (.5*m/l)*sqrt(4*m*l*y+m*m*(y**2))
    u=rand(j_,1)

    index=where(u>m/(x+m))
    x[index]=m*m/x[index]
    return x
