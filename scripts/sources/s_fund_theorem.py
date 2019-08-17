#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# # s_fund_theorem [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fund_theorem&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ftheoasnum).

# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from arpym.statistics import simulate_normal
from arpym.pricing import numeraire_mre
from arpym.tools import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem-implementation-step00): Upload data

# +

path = '../../../databases/temporary-databases/'

db_vpayoff = pd.read_csv(path+'db_valuation_vpayoff.csv', index_col=0)
v_payoff = db_vpayoff.values
db_vtnow = pd.read_csv(path+'db_valuation_vtnow.csv', index_col=0)
v_tnow = db_vtnow.values.T[0]
db_prob = pd.read_csv(path+'db_valuation_prob.csv', index_col=0)
p = db_prob.values.T[0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem-implementation-step01): Compute the minimum relative entropy numeraire probabilities

# +
p_mre, _ = numeraire_mre(v_payoff, v_tnow, p=p, k=1)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem-implementation-step02): Compute the left- and right-hand side of the fundamental theorem of asset pricing values

# +
x = v_tnow / v_tnow[1]
y = p_mre * (v_payoff[:, 1]**(-1))@v_payoff
# -

# ## Plots

# +
# initialize figure
fig = plt.figure()

plt.style.use('arpm')
plt.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'r')
plt.scatter(x, y, marker='x')
plt.axis([np.min(x), np.max(x), np.min(x), np.max(x)])
plt.xlabel('l. h. side')
plt.ylabel('r. h. side')
plt.legend(['identity line'])
add_logo(fig)
