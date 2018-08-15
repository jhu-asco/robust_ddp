#!/usr/bin/env python
# Propagate the ellipsoids linearly
import numpy as np
import scipy

def propagateEllipsoid(Sigma_i, Sigma_w_i, dynamics_params, feedback_gain):
    """
    Given an ellipsoid at 
    """
    A, B, G = dynamics_params
    if len(B.shape) == 1:
      Abar = A + np.outer(B, feedback_gain)
    else:
      Abar =  A + np.dot(B, feedback_gain)
    Sigma_Abar = np.dot(Sigma_i, Abar.T)
    Sigmaw_G = np.dot(Sigma_w_i, G.T)
    Cov_n = np.dot(Sigma_Abar.T, Sigma_Abar) + np.dot(Sigmaw_G.T, Sigmaw_G)
    w,v = scipy.linalg.eigh(Cov_n)
    Sigma_n = np.dot(v, np.dot(np.diag(np.sqrt(w)), v.T))
    return Sigma_n
