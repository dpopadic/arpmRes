from numpy import min as npmin
from numpy import std, zeros, sort, argsort, cumsum, abs, mean, exp, r_
from numpy.linalg import norm as linalgnorm
from scipy.stats import norm


def GHCalibration(epsi, FP, Tolerance, Da0, Db0, Dg0, Dh0, MaxItex):
    # ## input
    # a,b,g,h :[vectors] (n_ x 1) parameters of GH distribution
    # SqDist  :[vector] (n_ x 1) square root of distance beetwen model and data
    # iter    :[vector] (n_ x 1) number of itarations
    #
    # # output
    # epsi            :[matrix] (n_ x t_end) time series of invariants
    # FP              :[vector] (1 x t_end) flexible probabilities
    # Tolerance       :[scalar]
    # Da0,Db0,Dg0,Dh0 :[scalars] step of local search of a,b,g,h respectively
    # maxitex         :[scalar] maximun number of iterations

    n_ = epsi.shape[0]
    yy = zeros((epsi.shape))
    uu = zeros((epsi.shape))

    for nn in range(n_):
        yy[nn, :], idx = sort(epsi[nn, :]), argsort(epsi[nn, :])
        fp = FP[0, idx]
        uu[nn, :] = cumsum(fp)

    yy = yy[:, 1:-1]
    uu = uu[:, 1:-1]
    n_ = yy.shape[0]

    aGH = zeros((n_, 1))
    bGH = zeros((n_, 1))
    gGH = zeros((n_, 1))
    hGH = zeros((n_, 1))
    SSqDist = zeros((n_, 1))
    iiter = zeros((n_, 1))

    for k in range(n_):
        # parameter
        a0 = mean(yy[k, :])
        b0 = std(yy[k, :])
        g0 = 1.0e-03
        h0 = 1.0e-03
        input = norm.ppf(uu[k, :], 0, 1)
        ##
        Q_y = a0 + b0 * ((1 / g0) * (exp(g0 * input) - 1) * exp(0.5 * h0 * input ** 2))
        SqDist = linalgnorm(Q_y - yy[k, :])
        SqDistfit = 0
        iter = 0

        ####=========================================================================
        while (abs((SqDist - SqDistfit)) > Tolerance) and iter < MaxItex:
            iter = iter + 1
            SqDistfit = SqDist
            ####=========================================================================
            aa = r_[(a0 - Da0), a0, (a0 + Da0)]
            bb = r_[(b0 - Db0), b0, (b0 + Db0)]
            gg = r_[(g0 - Dg0), g0, (g0 + Dg0)]
            hh = r_[(h0 - Dh0), h0, (h0 + Dh0)]
            ##
            DistQ_y = zeros((len(aa), len(bb), len(gg), len(hh)))
            for ka in range(len(aa)):
                for kb in range(len(bb)):
                    for kg in range(len(gg)):
                        for kh in range(len(hh)):
                            Q_y = aa[ka] + bb[kb] * (1 / gg[kg]) * (exp(gg[kg] * input) - 1) * exp(
                                0.5 * hh[kh] * input ** 2)
                            DistQ_y[ka, kb, kg, kh] = linalgnorm(Q_y - yy[k, :])

            for ka in range(len(aa)):
                for kb in range(len(bb)):
                    for kg in range(len(gg)):
                        for kh in range(len(hh)):
                            if DistQ_y[ka, kb, kg, kh] == npmin(DistQ_y):
                                SqDist = npmin(DistQ_y)
                                aka = ka
                                bkb = kb
                                gkg = kg
                                hkh = kh
                                break

            a0 = aa[aka]
            b0 = bb[bkb]
            g0 = gg[gkg]
            h0 = hh[hkh]

            if h0 < 0:
                h0 = 0
                Dh0 = Dh0 / 100

            if g0 < 0:
                g0 = 0
                Dg0 = Dg0 / 100

            if b0 < 0:
                b0 = b0 + Db0
                Db0 = Db0 / 100
        aGH[k] = a0
        bGH[k] = b0
        gGH[k] = g0
        hGH[k] = h0
        SSqDist[k] = SqDist
        iiter[k] = iter
    return aGH, bGH, gGH, hGH, SSqDist, iiter

