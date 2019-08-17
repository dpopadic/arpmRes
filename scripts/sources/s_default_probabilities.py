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

# # s_default_probabilities [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_default_probabilities&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-supervised-machine-learning).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, \
                                                            QuantileTransformer
from sklearn import tree
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from arpym.tools import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-parameters)

test_size = 0.2  # proportion of the test set
n_sample = 10000  # num. of samples in the database; set =30000 to catch it all
pol_degree = 2  # degrees in polynomial features
lambda_lasso = 0.05  # lasso parameter
max_depth_tree = 10  # maximum depth of decision tree classifier
cross_val = 0  # set "1" to do cross-validation (computational time increases)
k_ = 5  # parameter of Stratified K-Folds cross-validator

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step00): Import data and pre-process database

# +
# Import data
path = '../../../databases/global-databases/credit/' + \
    'db_default_data_creditcardsclients/'
df = pd.read_csv(path+'db_default_data_creditcardsclients.csv')
df = df.iloc[:, 1:df.shape[1]]  # exlude ID

# Sort database so that the categorical features are at the beginning

# indexes of the categorical features
ind_cat = np.r_[np.arange(1, 4), np.arange(5, 11)]
n_cat = len(ind_cat)  # number of categorical features
# indexes of the continuous features
ind_cont = np.r_[np.array([0, 4]), np.arange(11, df.shape[1])]
n_cont = len(ind_cont)  # number of categorical features
df = df.iloc[:n_sample, np.r_[ind_cat, ind_cont]]

# Outputs and features
z = np.array(df.iloc[:, :-1])  # features
x = np.array(df.iloc[:, -1])  # labels

# Standardize continuous features
quantile_transformer = QuantileTransformer(output_distribution='normal')
z_cont = quantile_transformer.fit_transform(z[:, -n_cont:])

# Transform categorical features via one-hot encoding
# shift up, because the OneHotEncoder takes only positive inputs
enc = OneHotEncoder()
z_cat = enc.fit_transform(np.abs(np.min(z[:, :n_cat], axis=0)) +
                          z[:, :n_cat]).toarray()

n_enc = z_cat.shape[1]  # number of encoded categorical features

z = np.concatenate((z_cat, z_cont), axis=1)

# Define test set and estimation set
z_estimation, z_test, x_estimation, x_test = train_test_split(z, x)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step01): Logistic regression on continuous features

# Set C = +infinity to have 0 Lasso parameter
lg = LogisticRegression(penalty='l1', C=10**5, class_weight='balanced')
lg = lg.fit(z_estimation[:, -n_cont:], x_estimation)  # fit the model
p_z_lg = lg.predict_proba(z_test[:, -n_cont:])[:, 1]  # predict the probs
cm_lg = confusion_matrix(x_test, lg.predict(z_test[:, -n_cont:]))  # conf. mat.
er_lg = -np.sum(np.log(p_z_lg))  # error
print('Logistic error: %1.4f' % er_lg)
# conditional scores
s_0_lg = logit(lg.predict_proba(z_test[:, -n_cont:])[
                    np.where(x_test == 0)[0], 1])
s_1_lg = logit(lg.predict_proba(z_test[:, -n_cont:])[
                    np.where(x_test == 1)[0], 1])

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step02): Add interactions to logistic regression

# +
# Add interactions
poly = PolynomialFeatures(degree=pol_degree)
z_estimation_inter = poly.fit_transform(z_estimation[:, -n_cont:])
z_test_inter = poly.fit_transform(z_test[:, -n_cont:])

# Set C = +infinity to have 0 Lasso parameter
lg_inter = LogisticRegression(penalty='l1', C=10**5, class_weight='balanced')
lg_inter = lg_inter.fit(z_estimation_inter, x_estimation)  # fit the model
p_z_inter = lg_inter.predict_proba(z_test_inter)[:, 1]  # pred. the probs.
cm_inter = confusion_matrix(x_test, lg_inter.predict(z_test_inter))
er_inter = -np.sum(np.log(p_z_inter))  # error
print('Logistic with interactions error: %1.4f' % er_inter)
# conditional scores
s_0_inter = logit(lg_inter.predict_proba(z_test_inter)[
                                                  np.where(x_test == 0)[0], 1])
s_1_inter = logit(lg_inter.predict_proba(z_test_inter)[
                                                  np.where(x_test == 1)[0], 1])
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step03): Add encoded categorical features to logistic regression

# +
z_enc_estimation = np.concatenate((z_estimation[:, :n_enc],
                                   z_estimation_inter), axis=1)
z_enc_test = np.concatenate((z_test[:, :n_enc], z_test_inter), axis=1)

# Set C = +infinity to have 0 Lasso parameter
lg_enc = LogisticRegression(penalty='l1', C=10**5, class_weight='balanced')
lg_enc = lg_enc.fit(z_enc_estimation, x_estimation)  # fit the model
p_z_enc = lg_enc.predict_proba(z_enc_test)[:, 1]  # pred. the probs.
cm_enc = confusion_matrix(x_test, lg_enc.predict(z_enc_test))
er_enc = -np.sum(np.log(p_z_enc))  # error
print('Logistic with interactions and categorical error: %1.4f' % er_enc)
# conditional scores
s_0_enc = logit(lg_enc.predict_proba(z_enc_test)[np.where(x_test == 0)[0], 1])
s_1_enc = logit(lg_enc.predict_proba(z_enc_test)[np.where(x_test == 1)[0], 1])
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step04): Add lasso regularization

lg_lasso = LogisticRegression(penalty='l1', C=10**5, class_weight='balanced')
lg_lasso = lg_lasso.fit(z_enc_estimation, x_estimation)  # fit the model
p_z_lasso = lg_lasso.predict_proba(z_enc_test)[:, 1]  # predict the probs.
cm_lasso = confusion_matrix(x_test, lg_lasso.predict(z_enc_test))  # conf. mat.
er_lasso = -np.sum(np.log(p_z_lasso))  # error
print('Logistic with lasso error: %1.4f' % er_lasso)
# conditional scores
s_0_lasso = logit(lg_lasso.predict_proba(z_enc_test)[
                    np.where(x_test == 0)[0], 1])
s_1_lasso = logit(lg_lasso.predict_proba(z_enc_test)[
                    np.where(x_test == 1)[0], 1])

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step05): CART classifier

tree_clf = tree.DecisionTreeClassifier(max_depth=max_depth_tree,
                                       class_weight='balanced')  # def. method
tree_clf = tree_clf.fit(z_enc_estimation, x_estimation)  # fit the model
p_z_tree = tree_clf.predict_proba(z_enc_test)[:, 1]  # predict the scores
cm_tree = confusion_matrix(x_test, tree_clf.predict(z_enc_test))  # conf. mat.
er_tree = (cm_tree[0, 1]/np.sum(x_test == 0) +
           cm_tree[1, 0]/np.sum(x_test == 1))  # error
print('CART classifier error: %1.4f' % er_tree)
# conditional scores
eps = 10**-5  # set threshold to avoid numerical noise in the logit function
p_0_tree = tree_clf.predict_proba(z_enc_test)[np.where(x_test == 0)[0], 1]
p_0_tree[p_0_tree < eps] = eps
p_0_tree[p_0_tree > 1-eps] = 1-eps
p_1_tree = tree_clf.predict_proba(z_enc_test)[np.where(x_test == 1)[0], 1]
p_1_tree[p_1_tree < eps] = eps
p_1_tree[p_1_tree > 1-eps] = 1-eps
s_0_tree = logit(p_0_tree)
s_1_tree = logit(p_1_tree)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step06): Add gradient boosting to CART classifier

boost_clf = GradientBoostingClassifier(max_depth=max_depth_tree)  # method
boost_clf = boost_clf.fit(z_enc_estimation, x_estimation)  # fit the model
p_z_boost = boost_clf.predict_proba(z_enc_test)[:, 1]  # predict the probs.
cm_boost = confusion_matrix(x_test, boost_clf.predict(z_enc_test))  # conf. mat
er_boost = (cm_boost[0, 1]/np.sum(x_test == 0) +
            cm_boost[1, 0]/np.sum(x_test == 1))  # error
print('CART classifier with gradient boosting error: %1.4f' % er_boost)
# conditional scores
s_0_boost = logit(boost_clf.predict_proba(z_enc_test)[
                np.where(x_test == 0)[0], 1])
s_1_boost = logit(boost_clf.predict_proba(z_enc_test)[
                np.where(x_test == 1)[0], 1])

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step07): Compute fpr, tpr and AUC on the test set

# +
# 1) Logistic
fpr_lg, tpr_lg, _ = roc_curve(x_test, p_z_lg)
auc_lg = auc(fpr_lg, tpr_lg)
print('Logistic AUC: %1.3f' % auc_lg)

# 2) Logistic with interactions
fpr_inter, tpr_inter, _ = roc_curve(x_test, p_z_inter)
auc_inter = auc(fpr_inter, tpr_inter)
print('Logistic with interactions AUC: %1.3f' % auc_inter)

# 3) Logistic with interactions and encoded categorical features
fpr_enc, tpr_enc, _ = roc_curve(x_test, p_z_enc)
auc_enc = auc(fpr_enc, tpr_enc)
print('Logistic with interactions and categorical AUC: %1.3f' % auc_enc)

# 4) Logistic lasso with interactions and encoded categorical features
fpr_lasso, tpr_lasso, _ = roc_curve(x_test, p_z_lasso)
auc_lasso = auc(fpr_lasso, tpr_lasso)
print('Logistic with lasso AUC: %1.3f' % auc_lasso)

# 5) CART classifier
fpr_tree, tpr_tree, _ = roc_curve(x_test, p_z_tree)
auc_tree = auc(fpr_tree, tpr_tree)
print('CART classifier AUC: %1.3f' % auc_tree)

# 6) Gradient boosting classifier
fpr_boost, tpr_boost, _ = roc_curve(x_test, p_z_boost)
auc_boost = auc(fpr_boost, tpr_boost)
print('Gradient boosting classifier AUC: %1.3f' % auc_boost)
# -

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_default_probabilities-implementation-step08): Choose best probabilistic and point predictors via cross-validation

if cross_val == 1:
    # Split the estimation set into training and validation sets for k-fold
    # cross-validation
    k_fold = StratifiedKFold(n_splits=k_)
    z_train = []
    z_train_inter = []
    z_train_enc = []
    x_train = []
    z_val = []
    z_val_inter = []
    z_val_enc = []
    x_val = []
    for train, val in k_fold.split(z_estimation, x_estimation):
        z_train.append(z_estimation[train])
        x_train.append(x_estimation[train])
        z_val.append(z_estimation[val])
        x_val.append(x_estimation[val])
    for train, val in k_fold.split(z_estimation_inter, x_estimation):
        z_train_inter.append(z_estimation_inter[train])
        z_val_inter.append(z_estimation_inter[val])
    for train, val in k_fold.split(z_enc_estimation, x_estimation):
        z_train_enc.append(z_enc_estimation[train])
        z_val_enc.append(z_enc_estimation[val])
    # Probabilistic
    cv_er_lg = []
    cv_er_lasso = []
    cv_er_inter = []
    cv_er_enc = []
    for k in range(k_):
        # Logistic
        p_cv_lg = lg.fit(z_train[k], x_train[k]).predict_proba(z_val[k])
        cv_er_lg.append(-np.sum(np.log(p_cv_lg)))

        # Lasso
        p_cv_lasso = lg_lasso.fit(z_train[k],
                                  x_train[k]).predict_proba(z_val[k])
        cv_er_lasso.append(-np.sum(np.log(p_cv_lasso)))

        # Interactions
        p_cv_inter = lg_inter.fit(z_train_inter[k],
                                  x_train[k]).predict_proba(z_val_inter[k])
        cv_er_inter.append(-np.sum(np.log(p_cv_inter)))

        # Encoded categorical
        p_cv_enc = lg_inter.fit(z_train_enc[k],
                                x_train[k]).predict_proba(z_val_enc[k])
        cv_er_enc.append(-np.sum(np.log(p_cv_enc)))

    cv_er_lg = np.mean(cv_er_lg)
    cv_er_lasso = np.mean(cv_er_lasso)
    cv_er_inter = np.mean(cv_er_inter)
    cv_er_enc = np.mean(cv_er_enc)

    # Point
    cv_er_tree = []
    cv_er_boost = []
    for k in range(k_):
        # Tree
        cm_tree_cv =\
            confusion_matrix(x_val[k],
                             tree_clf.fit(z_train[k],
                                          x_train[k]).predict(z_val[k]))
        er_tree_cv = (cm_tree_cv[0, 1]/np.sum(x_val[k] == 0) +
                      cm_tree_cv[1, 0]/np.sum(x_val[k] == 1))  # error
        cv_er_tree.append(er_tree_cv)

        # Gradient boosting
        cm_boost_cv =\
            confusion_matrix(x_val[k],
                             boost_clf.fit(z_train[k],
                                           x_train[k]).predict(z_val[k]))
        er_boost_cv = (cm_boost_cv[0, 1]/np.sum(x_val[k] == 0) +
                       cm_boost_cv[1, 0]/np.sum(x_val[k] == 1))  # error
        cv_er_boost.append(er_boost_cv)

    cv_er_tree = np.mean(cv_er_tree)
    cv_er_boost = np.mean(cv_er_boost)

    print('Logistic CV error: %1.3f' % cv_er_lg)
    print('Logistic with interactions CV error: %1.3f' % cv_er_inter)
    print('Logistic with interactions and categorical CV error: %1.3f' %
          cv_er_enc)
    print('Logistic with lasso CV error: %1.3f' % cv_er_lasso)
    print('CART classifier CV error: %1.3f' % cv_er_tree)
    print('CART classifier with gradient boosting CV error: %1.3f' %
          cv_er_boost)

# ## Plots

plt.style.use('arpm')

# ## 1) Logistic regression

# +
fig1 = plt.figure()
ax11 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax12 = plt.subplot2grid((2, 2), (0, 1))
ax13 = plt.subplot2grid((2, 2), (1, 1))

# out of sample ROC curve
plt.sca(ax11)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.plot([0, 0, 1], [0, 1, 1], 'g')
plt.plot(fpr_lg, tpr_lg, 'b')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Random fit', 'Perfect fit', 'ROC curve'])
plt.text(0.05, 0.8, 'AUC = %.2f' % auc_lg)
plt.text(0.05, 0.85, 'Error = %.2f' % er_lg)
plt.title('Logistic regression (test set)')

# Scores
plt.sca(ax12)
plt.hist(s_0_lg, 80, density=True, alpha=0.7, color='r')
plt.hist(s_1_lg, 80, density=True, alpha=0.7, color='b')
plt.legend(['S | 0', 'S | 1'])
plt.title('Scores distribution')

# Confusion matrix
plt.sca(ax13)
cax_1 = plt.bar([0, 1], [cm_lg[0, 1]/np.sum(x_test == 0),
                         cm_lg[1, 0]/np.sum(x_test == 1)])
plt.ylim([0, 1.1])
plt.xticks([0, 1], ('$fpr$', '$fnr$'))
plt.title('Confusion matrix')
add_logo(fig1, location=1, size_frac_x=1/8)
plt.tight_layout()
# -

# ## 2) Logistic regression with interactions

# +
fig2 = plt.figure()
ax31 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax32 = plt.subplot2grid((2, 2), (0, 1))
ax33 = plt.subplot2grid((2, 2), (1, 1))

# out of sample ROC curve
plt.sca(ax31)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.plot([0, 0, 1], [0, 1, 1], 'g')
plt.plot(fpr_inter, tpr_inter, 'b')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Random fit', 'Perfect fit', 'ROC curve'])
plt.text(0.05, 0.8, 'AUC = %.2f' % auc_inter)
plt.text(0.05, 0.85, 'Error = %.2f' % er_inter)
plt.title('Logistic regression with interactions deg. = %1i (test set)'
          % pol_degree)

# Scores
plt.sca(ax32)
plt.hist(s_0_inter, 80, density=True, alpha=0.7, color='r')
plt.hist(s_1_inter, 80, density=True, alpha=0.7, color='b')
plt.legend(['S | 0', 'S | 1'])
plt.title('Scores distribution')

# Confusion matrix
plt.sca(ax33)
cax_1 = plt.bar([0, 1], [cm_inter[0, 1]/np.sum(x_test == 0),
                         cm_inter[1, 0]/np.sum(x_test == 1)])
plt.ylim([0, 1.1])
plt.xticks([0, 1], ('$fpr$', '$fnr$'))
plt.title('Confusion matrix')

add_logo(fig2, location=1, size_frac_x=1/8)
plt.tight_layout()
# -

# ## 3) Logistic regression with interactions and encoded categorical features

# +
fig3 = plt.figure()
ax21 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax22 = plt.subplot2grid((2, 2), (0, 1))
ax23 = plt.subplot2grid((2, 2), (1, 1))

# out of sample ROC curve
plt.sca(ax21)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.plot([0, 0, 1], [0, 1, 1], 'g')
plt.plot(fpr_enc, tpr_enc, 'b')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Random fit', 'Perfect fit', 'ROC curve'])
plt.text(0.05, 0.8, 'AUC = %.2f' % auc_enc)
plt.text(0.05, 0.85, 'Error = %.2f' % er_enc)
plt.title('Logistic regression with interactions and categorical features')

# Scores
plt.sca(ax22)
plt.hist(s_0_enc, 80, density=True, alpha=0.7, color='r')
plt.hist(s_1_enc, 80, density=True, alpha=0.7, color='b')
plt.legend(['S | 0', 'S | 1'])
plt.title('Scores distribution')

# Confusion matrix
plt.sca(ax23)
cax_1 = plt.bar([0, 1], [cm_enc[0, 1]/np.sum(x_test == 0),
                         cm_enc[1, 0]/np.sum(x_test == 1)])
plt.ylim([0, 1.1])
plt.xticks([0, 1], ('$fpr$', '$fnr$'))
plt.title('Confusion matrix')

add_logo(fig3, location=1, size_frac_x=1/8)
plt.tight_layout()
# -

# ## 4) Logistic regression with lasso

# +
fig4 = plt.figure()
ax21 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax22 = plt.subplot2grid((2, 2), (0, 1))
ax23 = plt.subplot2grid((2, 2), (1, 1))

# out of sample ROC curve
plt.sca(ax21)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.plot([0, 0, 1], [0, 1, 1], 'g')
plt.plot(fpr_lasso, tpr_lasso, 'b')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Random fit', 'Perfect fit', 'ROC curve'])
plt.text(0.05, 0.8, 'AUC = %.2f' % auc_lasso)
plt.text(0.05, 0.85, 'Error = %.2f' % er_lasso)
plt.title('Logistic regression with Lasso param. = %1.2e (test set)' %
          lambda_lasso)

# Scores
plt.sca(ax22)
plt.hist(s_0_lasso, 80, density=True, alpha=0.7, color='r')
plt.hist(s_1_lasso, 80, density=True, alpha=0.7, color='b')
plt.legend(['S | 0', 'S | 1'])
plt.title('Scores distribution')

# Confusion matrix
plt.sca(ax23)
cax_1 = plt.bar([0, 1], [cm_lasso[0, 1]/np.sum(x_test == 0),
                         cm_lasso[1, 0]/np.sum(x_test == 1)])
plt.ylim([0, 1.1])
plt.xticks([0, 1], ('$fpr$', '$fnr$'))
plt.title('Confusion matrix')

add_logo(fig4, location=1, size_frac_x=1/8)
plt.tight_layout()
# -

# ## 5) CART classifier

# +
fig5 = plt.figure()
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))

# out of sample ROC curve
plt.sca(ax1)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.plot([0, 0, 1], [0, 1, 1], 'g')
plt.plot(fpr_tree, tpr_tree, 'b')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Random fit', 'Perfect fit', 'ROC curve'])
plt.text(0.05, 0.8, 'AUC = %.2f' % auc_tree)
plt.text(0.05, 0.85, 'Error = %.2f' % er_tree)
plt.title('CART classifier: max. depth of tree = %1i (test set)'
          % max_depth_tree)

# Scores
plt.sca(ax2)
plt.hist(s_0_tree[~np.isinf(s_0_tree)], 80, density=True, alpha=0.7, color='r')
plt.hist(s_1_tree[~np.isinf(s_1_tree)], 80, density=True, alpha=0.7, color='b')
plt.legend(['S | 0', 'S | 1'])
plt.title('Scores distribution')

# Confusion matrix
plt.sca(ax3)
cax_1 = plt.bar([0, 1], [cm_tree[0, 1]/np.sum(x_test == 0),
                         cm_tree[1, 0]/np.sum(x_test == 1)])
plt.ylim([0, 1.1])
plt.xticks([0, 1], ('$fpr$', '$fnr$'))
plt.title('Confusion matrix')

add_logo(fig5, location=1, size_frac_x=1/8)
plt.tight_layout()
# -

# ## Decision regions

# +
fig6 = plt.figure()
# Parameters
n_classes = 2
plot_colors = "rb"
plot_step = 0.2

k1 = -10
k2 = -12

z_k1_min = z_estimation[:, k1].min()
z_k1_max = z_estimation[:, k1].max()
z_k2_min = z_estimation[:, k2].min()
z_k2_max = z_estimation[:, k2].max()
zz_k1, zz_k2 = np.meshgrid(np.arange(z_k1_min, z_k1_max, plot_step),
                           np.arange(z_k2_min, z_k2_max, plot_step))
tree_clf_plot = tree.DecisionTreeClassifier(max_depth=max_depth_tree,
                                            class_weight='balanced')
p_plot = tree_clf_plot.fit(z_estimation[:, [k1, k2]],
                           x_estimation).predict_proba(np.c_[zz_k1.ravel(),
                                                       zz_k2.ravel()])[:, 1]
p_plot = p_plot.reshape(zz_k1.shape)
cs = plt.contourf(zz_k1, zz_k2, p_plot, cmap=plt.cm.RdYlBu)

for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(x_estimation == i)
    plt.scatter(z_estimation[idx, k1], z_estimation[idx, k2], c=color,
                label=['0', '1'][i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.xlabel(list(df)[k1])
plt.ylabel(list(df)[k2])
plt.xlim([z_k1_min, z_k1_max])
plt.ylim([z_k2_min, z_k2_max])
plt.title('CART classifier decision regions')
add_logo(fig6, alpha=0.8, location=3)
plt.tight_layout()
# -

# ## 6) Gradient boosting classifier

# +
fig7 = plt.figure()
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))

# out of sample ROC curve
plt.sca(ax1)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.plot([0, 0, 1], [0, 1, 1], 'g')
plt.plot(fpr_boost, tpr_boost, 'b')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Random fit', 'Perfect fit', 'ROC curve'])
plt.text(0.05, 0.8, 'AUC = %.2f' % auc_tree)
plt.text(0.05, 0.85, 'Error = %.2f' % er_tree)
plt.title('CART classifier with gradient boosting (test set)')

# Scores
plt.sca(ax2)
plt.hist(s_0_boost, 80, density=True, alpha=0.7, color='r')
plt.hist(s_1_boost, 80, density=True, alpha=0.7, color='b')
plt.legend(['S | 0', 'S | 1'])
plt.title('Scores distribution')

# Confusion matrix
plt.sca(ax3)
cax_1 = plt.bar([0, 1], [cm_boost[0, 1]/np.sum(x_test == 0),
                         cm_boost[1, 0]/np.sum(x_test == 1)])
plt.ylim([0, 1.1])
plt.xticks([0, 1], ('$fpr$', '$fnr$'))
plt.title('Confusion matrix')

add_logo(fig7, location=1, size_frac_x=1/8)
plt.tight_layout()
# -

# ## Decision regions

# +
fig8 = plt.figure()
# Parameters
n_classes = 2
plot_colors = "rb"
plot_step = 0.2

k1 = -10
k2 = -12

z_k1_min = z_estimation[:, k1].min()
z_k1_max = z_estimation[:, k1].max()
z_k2_min = z_estimation[:, k2].min()
z_k2_max = z_estimation[:, k2].max()
zz_k1, zz_k2 = np.meshgrid(np.arange(z_k1_min, z_k1_max, plot_step),
                           np.arange(z_k2_min, z_k2_max, plot_step))
boost_clf_plot = GradientBoostingClassifier()
p_plot = boost_clf_plot.fit(z_estimation[:, [k1, k2]],
                            x_estimation).predict_proba(np.c_[zz_k1.ravel(),
                                                        zz_k2.ravel()])[:, 1]
p_plot = p_plot.reshape(zz_k1.shape)
cs = plt.contourf(zz_k1, zz_k2, p_plot, cmap=plt.cm.RdYlBu)

for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(x_estimation == i)
    plt.scatter(z_estimation[idx, k1], z_estimation[idx, k2], c=color,
                label=['0', '1'][i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.xlabel(list(df)[k1])
plt.ylabel(list(df)[k2])
plt.xlim([z_k1_min, z_k1_max])
plt.ylim([z_k2_min, z_k2_max])
plt.title('CART classifier with gradient boosting decision regions')
add_logo(fig8, alpha=0.8, location=3)
plt.tight_layout()
