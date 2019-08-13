from numpy import log, exp, r_

from MinRelEntFP import MinRelEntFP


def PathMomMatch(p, x, mu_k, sig2_k, pdfconstr):
    ## Matches first and second moments along the path, twisting probabilities via EP
    #  INPUTS
    #   x          [matrix]: (k_bar x j_bar) Monte Carlo scenarios
    #   p          [vector]: (1 x j_bar) Monte Carlo probabilities
    #   mu_k       [vector]: (k_bar x 1) constraints on means
    #   sig2_k     [vector]: (k_bar x 1) constraints on variances
    #   pdfconstr  [vector]: (1 x j_bar) constraints on marginal pdf probabilities

    #  OPS
    #   pbar_j [vector]: (j_bar x 1) twisted probabilities
    #   r      [scalar]: Relative Number of Effective Scenario

    # Flexible probabilities
    pbar = MinRelEntFP(p, None, None, r_[x, x**2, pdfconstr],  r_[mu_k, mu_k**2 + sig2_k, [[1]]])[0]

    # Relative Number of Effective Scenarios
    j_bar = x.shape[1]
    r = exp(-pbar@log(pbar.T))/j_bar
    return pbar, r
