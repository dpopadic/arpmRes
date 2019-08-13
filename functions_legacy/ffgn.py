import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, ones, zeros, real, conj, diff, abs, sqrt, r_, seterr
from numpy import min as npmin
from numpy.fft import ifft, fft
from numpy.random import randn

seterr(invalid='ignore')

plt.style.use('seaborn')


def ffgn(H, n, N):
    # Written jointly by Yingchun Zhou (Jasmine), zhouyc@math.bu.edu
    # and Stilian Stoev, sstoev@umich.edu, September, 2005.
    # abridged by A. Meucci: all credit to Yingchun and Stilian
    #
    # Generates exact paths of Fractional Gaussian Noise by using
    # circulant embedding (for 1/2<H<1) and Lowen's method (for 0<H<1/2).
    #
    # Input:
    #   H     <- Hurst exponent
    #   n     <- number of independent paths
    #   N     <- the length of the time series

    if 0.5 < H < 1:
        # Use the "circulant ebedding" technique.  This method works only in the case when 1/2 < H < 1.

        # First step: specify the covariance
        c_1 = abs(pow(arange(-1, N), (2 * H)))
        c_1[0] = 1
        c = 1 / 2 * (arange(1, N + 2) ** (2 * H) - 2 * (arange(N + 1) ** (2 * H)) + c_1)
        v = r_[c[:N], c[N:0:-1]]

        # Second step: calculate Fourier transform of c
        g = real(fft(v))

        if npmin(g) < 0:
            raise ValueError('Some of the g[k] are negative!')
        g = abs(g).reshape(1, -1)
        z = zeros((n, N + 1), dtype=np.complex128)
        y = zeros((n, N + 1), dtype=np.complex128)
        # Third step: generate {z[0],...,z( 2*N)}
        z[:, [0]] = sqrt(2) * randn(n, 1)
        y[:, 0] = z[:, 0]
        z[:, [N]] = sqrt(2) * randn(n, 1)
        y[:, N] = z[:, N].copy()
        a = randn(n, N - 1)
        b = randn(n, N - 1)
        z1 = a + b * 1j
        z[:, 1:N] = z1
        y1 = z1
        y[:, 1:N] = y1
        y = r_['-1', y, conj(y[:, N - 1:0:-1])]
        y = y * (ones((n, 1)) @ sqrt(g))

        # Fourth step: calculate the stationary process f
        f = real(fft(y)) / sqrt(4 * N)
        f = f[:, :N]
    elif H == 0.5:
        f = randn((n, N))
    elif (H < 0.5) & (H > 0):
        # Use Lowen's method for this case.  Copied from the code "fftfgn.m"

        G1 = randn((n, N - 1))
        G2 = randn((n, N - 1))
        G = (G1 + sqrt(-1) @ G2) / sqrt(2)
        GN = randn((n, 1))
        G0 = zeros((n, 1))
        H2 = 2 * H
        R = (1 - (arange(1, N) / N) ** H2)
        R = r_[1, R, 0, R, arange(N - 1, 1 + -1, -1)]
        S = ones((n, 1)) @ (abs(fft(R, 2 * N)) ** .5)
        X = r_[zeros((n, 1)), G, GN, conj(G[:, range(N - 1, 0, -1)])] * S
        x = ifft(X.T, 2 * N).T
        y = sqrt(N) @ real((x[:, :N] - x[:, 0] @ ones((1, N))))
        f = N ** H @ [y[:, 0], diff(y.T).T]
    else:
        raise ValueError('The value of the Hurst parameter H must be in (0,1) and was %.f' % H)

    return f