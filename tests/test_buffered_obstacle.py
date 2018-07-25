#!/usr/bin/env python

from robust_ddp.obstacle_with_buffer import BufferedSphericalObstacle
from scipy import optimize
import numpy as np
import numpy.testing as np_testing
import unittest

class TestBufferedObstacle(unittest.TestCase):
    def setUp(self):
        self.n = 3  # length of state
        self.center = np.array([1, 2])
        self.radius = 2.0
        self.obstacle = BufferedSphericalObstacle(self.center, self.radius)
        self.Sigma = np.eye(self.n)
        BufferedSphericalObstacle.set_buffer_ellipsoid(self.Sigma)

    def testBufferEllipsoid(self):
        obstacle_new = BufferedSphericalObstacle(np.zeros(2), 5)
        self.assertEqual(obstacle_new.buffer_ellipsoid,
                         self.obstacle.buffer_ellipsoid)

    def test_ProjectionBufferEllipsoid(self):
        Sigma = np.diag([1,2,3])
        BufferedSphericalObstacle.set_buffer_ellipsoid(Sigma)
        np_testing.assert_almost_equal(
            BufferedSphericalObstacle.buffer_ellipsoid[0],
            np.array([[0, 1], [1, 0]]))
        np_testing.assert_almost_equal(
            BufferedSphericalObstacle.buffer_ellipsoid[1],
            np.array([2, 1]))

    def test_mapStateJacobian(self):
        np_testing.assert_almost_equal(
            self.obstacle.mapStateJacobian(np.array([1,3,4])),
            np.array([[1, 0, 0], [0, 1, 0]]))

    def test_distance(self):
        BufferedSphericalObstacle.set_buffer_ellipsoid(np.zeros((self.n, self.n)))
        self.assertEqual(self.obstacle.distance(np.array([10, 0, 0]))[0], 0.0)
        self.assertEqual(self.obstacle.distance(np.array([3, 2, 3]))[0], 0.0)
        self.assertLess(self.obstacle.distance(np.array([1, 1, 3]))[0], 0.0)
        BufferedSphericalObstacle.set_buffer_ellipsoid(np.diag([5, 3, 10]))
        self.assertEqual(self.obstacle.distance(np.array([1, -3, 0]))[0], 0.0)
        self.assertEqual(self.obstacle.distance(np.array([1, -4, 0]))[0], 0.0)
        self.assertLess(self.obstacle.distance(np.array([1, -2, 0]))[0], 0.0)
        self.assertLess(self.obstacle.distance(np.array([1, 2, 0]))[0], 0.0)


    def test_distance_jacobian(self):
        distance_fun = lambda x : self.obstacle.distance(x)[0]
        x = np.random.sample(3)
        grad = optimize.approx_fprime(x, distance_fun, 1e-6)
        dist, jac = self.obstacle.distance(x, True)
        np_testing.assert_almost_equal(jac, grad, decimal=4)

if __name__=="__main__":
    unittest.main()
