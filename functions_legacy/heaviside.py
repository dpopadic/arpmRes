from numpy import sign

def heaviside(x):
    return 0.5 * (sign(x) + 1)