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

# # s_spectral_theorem [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_spectral_theorem&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=doc-s_spectral_theorem).

# +
import numpy as np

from arpym.tools import pca_cov
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-parameters)

s2 = np.array([[3.0, np.sqrt(2.0)],
              [np.sqrt(2.0), 2.0]])  # symmetric positive (semi)definite 2x2 matrix
lambda2_1 = 1.0  # first candidate eigenvalue
e_1 = np.array([1, -np.sqrt(2)])  # first candidate eigenvector
lambda2_2 = 4.0  # second candidate eigenvalue
e_2 = np.array([np.sqrt(2), 1])  # second candidate eigenvector
v = np.array([2.0/3.0, 1.0/3.0])  # test vector

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step01): Test the eigenvectors and eigenvalues

# +
is_eig_1 = np.allclose(np.matmul(s2, e_1), lambda2_1*e_1)
is_eig_2 = np.allclose(np.matmul(s2, e_2), lambda2_2*e_2)
print((lambda2_1, e_1), 'is an eigenvalue/eigenvector pair:', is_eig_1)
print((lambda2_2, e_2), 'is an eigenvalue/eigenvector pair:', is_eig_2)

# if inputs aren't eigenvalue/eigenvector pairs, calculate
if not(is_eig_1 and is_eig_2):
    # check s2 is symmetric and positive (semi)definite (Sylvester's criterion)
    if np.allclose(s2[0][1], s2[1][0]) \
    and np.linalg.det(s2) >= 0 and s2[0][0] >= 0:
        # calculate eigenvalues and eigenvectors
        eigvecs, eigvals = pca_cov(s2)
        lambda2_1 = eigvals[0]
        e_1 = eigvecs[:, 0]
        lambda2_2 = eigvals[1]
        e_2 = eigvecs[:, 1]
        is_eig_new_1 = np.allclose(np.matmul(s2, e_1), lambda2_1*e_1)
        is_eig_new_2 = np.allclose(np.matmul(s2, e_2), lambda2_2*e_2)
        print((lambda2_1, e_1), 'is an eigenvalue/eigenvector pair:',
              is_eig_new_1)
        print((lambda2_2, e_2), 'is an eigenvalue/eigenvector pair:',
              is_eig_new_2)
    else:
        print('s2 must be positive and symmetric')
        print('Make sure s2[0][1]=s2[1][0], s2[0][0]>=0 and np.linalg.det(s2)>=0')
        print('Determinant:', np.linalg.det(s2))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step02): Sort (eigenvalue, eigenvector) pairs by decreasing eigenvalue

# put the eigenvalue/eigenvector pairs into a list
spect_decomp = [[lambda2_1, e_1], [lambda2_2, e_2]]
# sort in decreasing order
spect_decomp.sort(reverse=True)
# update eigenvalue/eignvector labels
lambda2_1 = spect_decomp[0][0]
e_1 = spect_decomp[0][1]
lambda2_2 = spect_decomp[1][0]
e_2 = spect_decomp[1][1]

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step03): Test that eigenvectors are orthogonal and normalize the eigenvectors

# test orthogonality
eigvec_orth = np.allclose(np.vdot(e_1, e_2), 0)
# normalize the eigenvectors
e_1 = e_1/np.linalg.norm(e_1)
e_2 = e_2/np.linalg.norm(e_2)
# test length of normalized eigenvectors
length_e1 = np.round(np.vdot(e_1, e_1), 3)
length_e2 = np.round(np.vdot(e_1, e_1), 3)
print(e_1, 'and', e_2, 'are orthogonal:', eigvec_orth)
print('length of ', e_1, ': ', length_e1, sep='')
print('length of ', e_2, ': ', length_e2, sep='')

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step04): Choose eigenvector signs so that the determinant is positive

if np.linalg.det(np.vstack((e_1, e_2)).T) < 0:
    e_2 = -e_2
# check this is still an eigenvector
is_neg_eig_2 = np.allclose(np.matmul(s2, e_2), lambda2_2*e_2)
print((lambda2_2, e_2), 'is an eigenvalue/eigenvector pair:', is_neg_eig_2)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step05): Calculate the eigenvalues/eigenvectors of s2 using recursive solution

# in two dimensions, change to polar coordinates and solve
theta = 0.5*np.arctan2(2*s2[0][1], (s2[0][0]-s2[1][1]))
e_1_calc = np.array([np.cos(theta), np.sin(theta)])
lambda2_1_calc = np.matmul(e_1_calc, np.matmul(s2, e_1_calc))
e_2_calc = np.array([np.sin(theta), -np.cos(theta)])
lambda2_2_calc = np.matmul(e_2_calc, np.matmul(s2, e_2_calc))
# check that these are the same as input/calculated up to sign of eigenvectors
is_eig_calc_1 = (np.allclose(np.abs(e_1_calc), np.abs(e_1)) and
                 np.allclose(lambda2_1_calc, lambda2_1))
is_eig_calc_2 = (np.allclose(np.abs(e_2_calc), np.abs(e_2)) and
                 np.allclose(lambda2_2_calc, lambda2_2))
print((lambda2_1_calc, e_1_calc),
      ' matches eigenvalue/eigenvector pair (up to sign) ',
      (lambda2_1, e_1), ': ', is_eig_calc_1, sep='')
print((lambda2_2_calc, e_2_calc),
      ' matches eigenvalue/eigenvector pair  (up to sign) ',
      (lambda2_2, e_2), ': ', is_eig_calc_2, sep='')

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step06): Put eigenvectors into a matrix

# define matrix
e = np.vstack((e_1, e_2)).T
# check to see if rotation
is_eigmat_rotation = np.allclose(np.linalg.det(e), 1)
print('The eigenvector matrix is a rotation:', is_eigmat_rotation)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step07): Verify the eigenvector matrix is orthogonal

is_eigmat_orth = np.allclose(np.matmul(e, e.T), np.identity(2))
print('The eigenvector matrix is orthogonal:', is_eigmat_orth)

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step08): Demonstrate effect of the eigenvector matrix on a vector

# multiply vector v by eigenvector matrix e
v_trans = np.matmul(e, v)
# calculate the square norms
is_norm_v_same = np.allclose(np.linalg.norm(v_trans)**2,
                             np.linalg.norm(v)**2)
print('The eigenvector matrix does not change the length of vector ',
      v, ': ', is_norm_v_same, sep='')

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step09): Create diagonal matrix of eigenvalue roots

diag_lambda = np.diag(np.array([np.sqrt(lambda2_1), np.sqrt(lambda2_2)]))

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step10): Verify spectral decomposition returns original matrix

spect_decomp_gives_s2 = np.allclose(
        np.matmul(e, (np.matmul(diag_lambda, np.matmul(diag_lambda, e.T)))),
        s2)
print('The spectral decomposition returns the original matrix s2:',
      spect_decomp_gives_s2)

# ## [Step 11](https://www.arpm.co/lab/redirect.php?permalink=s_spectral_theorem-implementation-step11): Verify trace and determinant identities

trace_is_sum_eigvals = np.allclose(np.trace(s2), lambda2_1+lambda2_2)
det_is_prod_eigvals = np.allclose(np.linalg.det(s2), lambda2_1*lambda2_2)
print('The trace of s2 equals the sum of the eigenvalues:',
      trace_is_sum_eigvals)
print('The determinant of s2 equals the product of the eigenvalues:',
      det_is_prod_eigvals)
