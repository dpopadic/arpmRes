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

# # s_rating_migrations [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_rating_migrations&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-rating-migrations).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.tools import aggregate_rating_migrations, add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_rating_migrations-parameters)

# start of period for aggregate credit risk drivers
tfirst_credit = np.datetime64('1995-01-01')
# end of period for aggregate credit risk drivers
tlast_credit = np.datetime64('2004-12-31')

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_rating_migrations-implementation-step00): Import data

# ratings
rating_path = '../../../databases/global-databases/credit/db_ratings/'
db_ratings = pd.read_csv(rating_path+'data.csv', parse_dates=['date'])
# ratings_param represents all possible ratings i.e. AAA, AA, etc.
ratings_param = pd.read_csv(rating_path+'params.csv', index_col=0)
ratings_param = np.array(ratings_param.index)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_rating_migrations-implementation-step01): Extract aggregate credit risk drivers

dates, n_obligors, n_cum_trans, _, n_tot, _ = \
    aggregate_rating_migrations(db_ratings, ratings_param, tfirst_credit,
                                tlast_credit)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_rating_migrations-implementation-step02): Save databases

# +
path = '../../../databases/temporary-databases/'

a, b = np.meshgrid(ratings_param, ratings_param)
col = ratings_param.tolist()
col = col + list(zip(a.reshape(-1),b.reshape(-1)))
out = pd.DataFrame(np.c_[n_obligors,
                         n_cum_trans.reshape(dates.shape[0],
                                             n_cum_trans.shape[1]*n_cum_trans.shape[2])],
                  index=dates, columns=col)
out.to_csv(path+'db_credit_rd.csv')
del out
# -

# ## Plots

# +
# plot 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.view_init(-42, 28)
n_plot = n_cum_trans[-1]
nx, ny = n_plot.shape
xpos,ypos = np.meshgrid(np.arange(nx), np.arange(ny))
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
xpos = xpos - 0.2
ypos = ypos - 0.5
zpos = np.zeros_like(xpos)
# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = n_plot.flatten()
ax.bar3d(xpos,ypos,zpos, dx,dy,dz,cmap='gray')  # width = 0.5
ax.set_title('Cumulative number of transitions')
ax.set_xlabel('From',labelpad=20)
ax.set_ylabel('o',labelpad=20)
ax.set_xlim([-1, 8])
ax.set_ylim([-1, 8])
ax.set_zlim([0, np.max(n_cum_trans[-1])])
plt.yticks(np.arange(8),[ratings_param[i] for i in range(8)], size='small')
plt.xticks(np.arange(8),[ratings_param[i] for i in range(8)], size='small')
plt.tight_layout();

# plot 2
f2, ax2 = plt.subplots(1, 1)
ax2.plot(dates, n_tot, '-b')
ax2.set_xlim([min(dates), max(dates)])
ax2.set_ylim([0, n_tot[-1]])
ax2.set_title('Total number of transitions: {transitions:.0f}'.format(transitions=n_tot[-1]));
