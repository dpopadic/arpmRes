import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('seaborn')


def BachelierCallPrice(x,s):
   # This function computes the zero-strike call option price profile
   # according to the Bachelier pricing function.

   # INPUTS:
   # x     :[vector] rate
   # s     :[scalar] smoothing parameter

   # OP:
   # C     :[scalar] inverse-call transformed rate

   C = x*norm.cdf(x/s)+s*norm.pdf(x/s)
   return C
