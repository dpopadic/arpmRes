from numpy import diag


def LSFP(y,x,p):
    #Least square with flexible probabilities Y=BX+e
    #dimensions:
    #y : [matrix] (n_ x t_end)
    #x : [matrix] (k_ x t_end)
    #p : [vector] (1 x t_end) flexible probabilities
    P=diag(p)
    b=(y@P@x.T)/(x@P@x.T)
    return b

