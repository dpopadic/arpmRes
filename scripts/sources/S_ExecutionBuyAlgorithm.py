#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # S_ExecutionBuyAlgorithm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_ExecutionBuyAlgorithm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-execution_-buy-algorithm-2).

# ## Prepare the environment

# +
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))

from numpy import array, sort, squeeze, \
    round, r_
from numpy.random import randint

import matplotlib.pyplot as plt

plt.style.use('seaborn')

# parameters
deltah_residual = array([80])  # shares of a given stock we want to buy
t_end = 60000  # end of the execution time interval in milliseconds
# -

# ## Randomly generate the ticks over the 1 munute time interval
k_ = 20  # effective number of ticks in the interval
time_of_ticks = r_[sort(randint(t_end + 1, size=(k_, 1)), 0), array(
    [[t_end]])]  # clock time at which ticks occur (tick times are uniformly distributed)

# +
# ## Proceed with the algorithm until the order is fully executed
time = 0  # initialize clock time
E_kleft = array([30])  # initialize the expectation on the number of ticks during the interval
k = 0  # initialize tick time
deltah_child = array([0])

while time < t_end - 1:  # clock time cycle
    time = time + 1
    if time == time_of_ticks[k]:  # a tick occurs
        deltah_child = r_[deltah_child, round(
            deltah_residual[k] / E_kleft[k])]  # compute the deltah_child size according to the algorithm

        E_kleft = r_[E_kleft, round((k + 1) * (
                    t_end - time) / time)]  # review the expectation on the residual tick time according to the proportion "k:time=E_kleft:time_left")
        deltah_residual = r_[
            deltah_residual, deltah_residual[k] - deltah_child[k]]  # compute the residual amount to be sold
        k = k + 1

        # if the residual amount is positive and the expected number of ticks left is positive proceed with the algo, otherwise stop
        if deltah_residual[k] <= 0 or E_kleft[k] == 0:
            break

        # ## Display the buy limit orders placed at each tick, showing also the corresponding clock time
# ## also show that limit orders are converted into market orders if within the next tick they have not been filled.

deltah_child = deltah_child[1:]
if deltah_residual[-1] > 0:
    for tick in range(len(deltah_child)):
        print('k = {tick} : place a limit deltah_child to buy {dtick} units at the {tot}th millisecond. If within the'
              ' {tot2}th millisecond the deltah_child has not been executed convert it into a market deltah_child to '
              'buy.'.format(tick=tick + 1,
                            dtick=deltah_child[tick],
                            tot=squeeze(time_of_ticks[tick]),
                            tot2=squeeze(time_of_ticks[tick + 1])
                            ))
    print('Place a market deltah_child to buy the remaining {dtick} units at the best ask at the end of the '
          'minute'.format(dtick=deltah_residual[-1]))
else:
    for tick in range(len(deltah_child)):
        print(
            'k = {tick} : place a limit deltah_child to buy {dtick} units at the {tot}th millisecond. If within the'
            ' {tot2}th millisecond the deltah_child has not been executed convert it into a market deltah_child to '
            'buy.'.format(tick=tick + 1,
                          dtick=deltah_child[tick],
                          tot=squeeze(time_of_ticks[tick]),
                          tot2=squeeze(time_of_ticks[tick + 1])
                          ))
