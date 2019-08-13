from collections import namedtuple

from numpy import ones, sort, argsort, diff, abs, mean
from numpy import sum as npsum, max as npmax


def ScoringAssessmentFP(score, x, p):
    #Scoring assessment measures with Flexible Probabilities
    # INPUT
    # score :[vector] (n_x 1) scores
    # x     :[vector] (n_x 1) default indicator (0/1)
    # p     :[vector] (n_x 1) flexible probabilities
    # OP
    # CdF   :[struct] with fields
    #                 s_bar: sorted scores
    #                 marginal: marginal cdf of the score
    #                 CondDefProb : conditional default cdf
    #                 CondSurvProb: conditional survival cdf
    # Index :[struct] with fields
    #                 AUC: area under the Lorentz curve
    #                 Gini: Gini coefficient
    #                 Gini_max: maximum possible theoretical value for Gini index
    #                 AR: accuracy ratio
    # indicator :[vector] (n_x 1) Bernoulli indicator

    ## Code

    if p is None:
        p = ones((len(x),1))/len(x)

    # number of obs
    n_ = len(score)

    # thresholds
    CdF = namedtuple('cdf',['s_bar','marginal','CondDefProb','CondSurvProb'])
    CdF.s_bar,i_s = sort(score), argsort(score)
    indicator = x[i_s]

    # compute marginal, conditional default and conditional survival
    # probabilities
    s = score[x==1]
    p_s = p[x==1]
    for j in range(1,n_):
        CdF.marginal[j] = npsum(score <= CdF.s_bar[j])
        index = s <= CdF.s_bar[j]
        CdF.CondDefProb[j] = npsum(p_s[index])

    CdF.marginal = CdF.marginal/npmax(CdF.marginal)
    CdF.CondDefProb = sort(CdF.CondDefProb/npsum(p*x))
    CdF.CondSurvProb = (CdF.marginal - CdF.CondDefProb*npsum(p*x))/(1 - npsum(p*x))

    # compute AUC, Gini and AR
    Index = namedtuple('index',['AUC','Gini','Gini_max','AR'])
    Index.AUC = npsum((1-CdF.CondDefProb[1:] + 1-CdF.CondDefProb[:-1])*abs(diff(1-CdF.marginal,1))/2)
    Index.Gini = 2*Index.AUC - 1
    Index.Gini_max = 1 - mean(x)
    Index.AR = Index.Gini/Index.Gini_max
    return CdF, Index, indicator
