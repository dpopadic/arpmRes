from numpy import zeros, exp


def PerpetualAmericanCall(x, varargin):
    # Price of a perpetual American call option with Bachelier underlying
    # INPUTS
    # x: underlying [vector]
    # varargin are either
    #   mu [scalar < 0]: drift parameter for the underlying
    #   sigma [scalar]: volatility parameter for the underlying
    #   k [scalar]: strike of the option (default k=0)
    # or
    #   eta [scalar]: inverse-call transformation parameter (=-sigma**2/(2mu)). Then k=0 by default.
    # OP
    # CallPrice [vector]

    if len(varargin.keys())==3:
        mu=varargin['mu']
        sigma=varargin['sigma']
        k=varargin['k']
        gamma=(-2*mu)/(sigma**2)
        eta=1/gamma

    elif len(varargin.keys())==2:
        mu=varargin['mu']
        sigma=varargin['sigma']
        k=0
        gamma=(-2*mu)/(sigma**2)
        eta=1/gamma

    elif len(varargin.keys())==1:
        k=0
        eta=varargin['eta']
        gamma=1/eta

    CallPrice=zeros((x.shape))

    boundary=k+eta

    CallPrice[x>=boundary]=x[x>=boundary]-k
    CallPrice[x<boundary]=eta*exp(-gamma*boundary)*exp(gamma*x[x<boundary])

    return CallPrice
