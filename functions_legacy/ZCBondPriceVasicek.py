from numpy import exp


def ZCBondPriceVasicek(tau, theta):
    # Computes the price of a zero-coupon bond according to the Vasicek model
    # INPUTS
    # tau  :[column vector] time to maturity
    # theta    :[struct] parameters of Vasicek model
    # OUTPUTS
    # z  :[column vector] bond prices corresponding to different times to maturity

    # For details on the exercise, see here .
    ## Code

    beta = (1 - exp(-theta.theta1 * tau)) / theta.theta1

    alpha = (theta.theta2 - ((theta.theta3 ** 2) / (2 * theta.theta1 ** 2))) * (beta - tau) - (
                (theta.theta3 ** 2) * (beta ** 2) / (4 * theta.theta1))

    z = exp(alpha - theta.theta0 * beta)
    return z
