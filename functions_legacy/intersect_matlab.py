from numpy import intersect1d, in1d, nonzero


def intersect(x, y):
    z = intersect1d(x, y)
    mx = in1d(x, y)
    ix = nonzero(mx)[0]
    my = in1d(y, x)
    iy = nonzero(my)[0]
    return z, ix, iy
