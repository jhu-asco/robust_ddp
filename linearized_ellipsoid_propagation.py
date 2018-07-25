#!/usr/bin/env python
# Propagate the ellipsoids linearly
import numpy as np

def propagateEllipsoid(Sigma_i, Sigma_w_i, dynamics_params, feedback_gain):
    """
    Given an ellipsoid at 
    """
    A, B, G = dynamics_params
    if len(B.shape) == 1:
      Abar = A + np.outer(B, feedback_gain)
    else:
      Abar =  A + np.dot(B, feedback_gain)
    Sigma_n = np.dot(Abar, np.dot(Sigma_i, Abar.T)) + np.dot(G, np.dot(Sigma_w_i, G.T))
    return Sigma_n
