import matplotlib.pyplot as plt
from numpy import zeros, sort, where, argsort, sqrt, sum as npsum
from numpy.linalg import solve, pinv
from scipy.stats import chi2

plt.style.use('seaborn')

from HighBreakdownFP import HighBreakdownFP


def DetectOutliersFP(epsi,p,q=0.975):
    #Outlier detection with Flexible Probabilities based on Mahalanobis distance
    # INPUTS
    # epsi         : [matrix] (i_ x t_end) observations - with zeros's for missing values
    # p            : [vector] (t_end x 1) flexible probabilities
    # q            : [scalar]  treshold. Observations with Mahalanobis distance from the estimated expectation (using the estimated covariance)
    #                          greater than F**{-1}[q], where F is the cdf of a chi distribution with i_ degrees of freedom, are detected as outliers
    # OUTPUTS
    # Positions      : [vector] (1 x number of outliers) Position of the outliers in descending order of distance
    # Outliers       : [matrix] (i_ x number of outliers) Outliers in descending order of distance
    # MahalDist      : [vector] (1 x number of outliers) Mahalanobis distances corresponding to Outliers

    # For details on the exercise, see here .

    ## Code

    i_,t_=epsi.shape

    #Step 1. Location/Dispersion (High Breakdown with Flexible Probabilities Ellipsoid)
    mu, sigma2, *_=HighBreakdownFP(epsi,p,1,0.75)

    #Step 2. Rescale dispersion
    sigma2=sigma2/chi2.ppf(0.75,i_)

    #Mahalanobis distances
    Mah = zeros(t_)
    for t in range(t_):
        Mah[t]=sqrt((epsi[:,t]-mu).T.dot(pinv(sigma2))@(epsi[:,t]-mu))

    #threshold
    threshold=sqrt(chi2.ppf(q,i_))

    #detect outliers
    Positions=where(Mah>=threshold)[0]
    Outliers=epsi[:,Positions]

    #Output outliers ordered by descending Mahalanobis distances
    MahalDist,index=sort(Mah[Positions])[::-1], argsort(Mah[Positions])[::-1]
    Positions=Positions[index]
    Outliers=Outliers[:,index]
    return Positions, Outliers, MahalDist
