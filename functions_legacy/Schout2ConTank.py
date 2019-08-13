from numpy import sqrt


def Schout2ConTank(a, b, d):
    # This function converts parameters from Schoutens notation to Cont-Tankov
    # notation

    ## Code
    th = d * b / sqrt(a ** 2 - b ** 2)
    k = 1 / (d * sqrt(a ** 2 - b ** 2))
    s = sqrt(d / sqrt(a ** 2 - b ** 2))
    return th, k, s
