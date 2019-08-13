from numpy import abs


def NormTradPLMeanVar(h_start,h_end,q_bar,alpha,beta,eta,gamma,sigma,delta_q):
    ## This function computes the expected value and variance of the normalized
    # trading P&L of the quasi-optimal execution strategy under the Almgren-Chriss
    # specification.
    #
    # INPUTS
    #   h_start :[vector](1 x j_) initial normalized position
    #   h_end   :[vector](1 x j_) terminal normalized position
    #   q_bar   :[scalar] average daily volume
    #   alpha   :[scalar] slippage power
    #   beta    :[scalar] acceleration parameter
    #   eta     :[scalar] normalized slippage coefficient
    #   gamma   :[scalar] normalized permanent impact coefficient
    #   sigma   :[scalar] normalized volatility
    #   delta_q :[scalar] volume time horizon
    # OUTPUTS
    #   m_Pi    :[vector](1 x j_) mean of the normalized trading P&L
    #   v_Pi    :[vector](1 x j_) variance of the normalized trading P&L

    # For details on the exercise, see here .
    ## Code

    xi = beta**(alpha+1)/(beta+beta*alpha-alpha)
    m_Pi = q_bar*(gamma/2*(h_end**2-h_start**2)-eta*xi*abs(h_end-h_start)**(1+alpha)*delta_q**(-alpha))
    v_Pi = (q_bar*sigma)**2*delta_q*(h_start**2+2*h_start*(h_end-h_start)/(beta+1)+(h_end-h_start)**2/(2*beta+1))

    return m_Pi, v_Pi
