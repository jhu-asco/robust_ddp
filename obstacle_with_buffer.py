#!/usr/bin/env python

from optimal_control_framework.costs.obstacle import SphericalObstacle
import numpy as np

class BufferedSphericalObstacle(SphericalObstacle):
    buffer_ellipsoid = None  # Principal axes (R.T), semi_major_axes
    n = None
    def __init__(self, center, radius):
        super(BufferedSphericalObstacle, self).__init__(center, radius)
        BufferedSphericalObstacle.n = self.n

    @classmethod
    def set_buffer_ellipsoid(self, Sigma):
        radii, principal_axes = np.linalg.eigh(Sigma)
        L = principal_axes.T*radii
        if self.n is None:
            self.n = radii.size
        U,Sigma,V = np.linalg.svd(L[:self.n, :])
        self.buffer_ellipsoid = (U, Sigma)

    def distance(self, x, compute_grads=False):
        if self.buffer_ellipsoid is None:
            BufferedSphericalObstacle.buffer_ellipsoid = (1, 0)
        principal_axes, semi_major_axes = self.buffer_ellipsoid
        scale = semi_major_axes + self.radius
        z = self.mapState(x)
        error = z - self.center
        error_scaling_mat = self.radius*(principal_axes/scale[:, np.newaxis])
        error_map = np.dot(error_scaling_mat, error)
        distance, jac = super(
            BufferedSphericalObstacle, self).distance_substep(
                error_map, compute_grads)
        if compute_grads:
            if distance < - self.tol:
                z_x = self.mapStateJacobian(x)
                error_map_x = np.dot(z_x.T, error_scaling_mat.T)
                jac = np.dot(error_map_x, jac)
            else:
                jac = np.zeros_like(x)
        return distance, jac
