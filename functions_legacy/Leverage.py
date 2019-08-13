from numpy import sum as npsum, zeros, abs


def Leverage(marketvalue, holdings, prices):
    #calculate leverage with transaction costs

    n_,t_=holdings.shape

    posExpos=marketvalue*(marketvalue>0)
    negExpos=marketvalue*(marketvalue<0)
    #transaction cost
    c=0.002
    transCost =zeros((n_,t_))
    transCost[:,0] =c*abs(holdings[:,0])*prices[:,0]
    transCost[:,1:]=c*abs(holdings[:,:-1]-holdings[:,1:])*prices[:,1:]
    dailyTransCost=npsum(transCost)
    NAV=5000000-dailyTransCost
    Lev=(abs(npsum(posExpos))**2+abs(npsum(negExpos))**2)/NAV
    return Lev
