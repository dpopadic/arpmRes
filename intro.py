import numpy as np

# simplest case as illustration
# -----------------------------

# 1. risk driver identification
v_1t = np.array([14.21, 14.29, 14.2, 14.29, 14.28, 14.26, 14.62, 14.45, 14.33, 14.35, 14.24])
v_2t = np.array([48.52, 48.55, 48.24, 48.13, 47.91, 48.22, 48.55, 48.18, 48.72, 48.52, 48.61])

# compute log values which represent the risk drivers
x_1t = np.log(v_1t)
x_2t = np.log(v_2t)


# 2. quest for invariance
e_1t = np.diff(x_1t)
e_2t = np.diff(x_2t)


# 3. estimation
mu = np.array([np.mean(e_1t), np.mean(e_2t)]) / 252
sigma = np.cov(np.array([e_1t, e_2t])) / 252

# 4. projection










