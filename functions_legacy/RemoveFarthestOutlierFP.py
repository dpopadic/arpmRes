from numpy import ones ,delete

from FarthestOutlier import FarthestOutlier


def RemoveFarthestOutlierFP(epsi,p,dates,p_remove=None):
    # This function removes the entry - corresponding to the worst outlier of
    # dataset epsi according to Flexible Probabilities p_remove - from the
    # arrays epsi, p and dates.
    # INPUTS
    #  epsi      :[matrix](n_ x t_end) dataset
    #  p         :[matrix](q_ x t_end) matrix containing q_ Flexible Probability profiles (not used for computing outliers)
    #  dates     :[vector](1 x t_end) vector of dates
    #  p_remove  :[vector](1 x t_end) vector of Flexible Probabilities aimed at computing the farthest outlier
    # OUTPUTS
    #  epsi      :[matrix](n_ x t_end-1) dataset of invariants with the farthest outlier deleted
    #  p         :[matrix](q_ x t_end-1) matrix of q_ Flexible Probability profiles with the entry corresponding to farthest outlier deleted
    #  dates     :[vector](1 x t_end-1) vector of dates with the entry corresponding to farthest outlier deleted

    ## Code
    t_=epsi.shape[1]

    if p_remove is None:
        p_remove=(1/t_)*ones((1,t_))

    t_tilde=FarthestOutlier(epsi,p_remove)
    epsi= delete(epsi,t_tilde,1)
    p= delete(p,t_tilde,1)
    dates= delete(dates,t_tilde,0)
    return epsi,p,dates
