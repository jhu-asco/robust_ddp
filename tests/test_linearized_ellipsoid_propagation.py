#!/usr/bin/env python

from robust_ddp.linearized_ellipsoid_propagation import propagateEllipsoid
import numpy as np
import numpy.testing as np_testing
import unittest

class TestEllipsoidPropagation(unittest.TestCase):
    def test_no_noise_identity_dynamics(self):
        n = 4
        m = 2
        dynamic_params = (np.eye(n), np.zeros((n, m)), np.eye(n))
        L = np.tril(np.random.sample((n,n)))
        Sigma_i = np.dot(L, L.T)
        Sigma_w_i = np.zeros((n,n))
        feedback_gain = np.random.sample((m,n))
        Sigma_n = propagateEllipsoid(Sigma_i, Sigma_w_i, dynamic_params,
                                     feedback_gain)
        np_testing.assert_almost_equal(Sigma_n, Sigma_i)

    def test_noise_identity_dynamics(self):
        n = 4
        m = 2
        dynamic_params = (np.eye(n), np.zeros((n, m)), np.eye(n))
        L = np.tril(np.random.sample((n,n)))
        Sigma_i = np.dot(L, L.T)
        Sigma_w_i = np.eye(n)
        feedback_gain = np.random.sample((m,n))
        Sigma_n = propagateEllipsoid(Sigma_i, Sigma_w_i, dynamic_params,
                                     feedback_gain)
        np_testing.assert_almost_equal(Sigma_n, Sigma_i+Sigma_w_i)

    def test_no_noise_stable_dynamics(self):
        n = 2
        m = 1
        dynamic_params = (np.eye(n), np.array([0, 1]), np.eye(n))
        Sigma_i = np.eye(n)
        Sigma_w_i = np.zeros((n,n))
        feedback_gain = np.array([-0.1, -0.1])
        Sigma_n = propagateEllipsoid(Sigma_i, Sigma_w_i, dynamic_params,
                                     feedback_gain)
        self.assertLess(Sigma_n[1,1], Sigma_i[1,1])
        self.assertEqual(Sigma_n[0,0], Sigma_i[0,0])

    def test_noise_stable_dynamics(self):
        n = 2
        m = 1
        dynamic_params = (np.eye(n), np.array([0, 1]), np.eye(n))
        Sigma_i = np.eye(n)
        Sigma_w_i = 0.5*np.eye(n)
        feedback_gain = np.array([-0.1, -0.1])
        Sigma_n = propagateEllipsoid(Sigma_i, Sigma_w_i, dynamic_params,
                                     feedback_gain)
        self.assertGreater(Sigma_n[1,1], Sigma_i[1,1])
        self.assertGreater(Sigma_n[0,0], Sigma_i[0,0])

