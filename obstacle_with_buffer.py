#!/usr/bin/env python

from optimal_control_framework.costs.obstacle import SphericalObstacle
import numpy as np

class BufferedSphericalObstacle(SphericalObstacle):
    buffer_ellipsoid = None  # Principal axes (R.T), semi_major_axes
    sigma_inflation = 1  # Inflate ellipsoids
    n = None
    def __init__(self, center, radius):
        super(BufferedSphericalObstacle, self).__init__(center, radius)
        BufferedSphericalObstacle.n = self.n

    @classmethod
    def set_buffer_ellipsoid(self, Sigma):
        radii, R = np.linalg.eigh(Sigma)
        L = R*radii
        if self.n is None:
            self.n = radii.size
        U,Sigma,V = np.linalg.svd(L[:self.n, :])
        self.buffer_ellipsoid = (U.T, Sigma)

    def distance(self, x, compute_grads=False):
        ebar, ebar_x_T = self.findError(x)
        if self.buffer_ellipsoid is None:
            error_map = ebar
            error_scaling_mat = np.eye(ebar_x_T.shape[1])
        else:
            principal_axes, semi_major_axes = self.buffer_ellipsoid
            scale = self.sigma_inflation*semi_major_axes + self.radius
            error_scaling_mat = self.radius*(principal_axes/scale[:, np.newaxis])
            error_map = np.dot(error_scaling_mat, ebar)
        distance, jac = super(
            BufferedSphericalObstacle, self).distance_substep(
                error_map, compute_grads)
        if compute_grads:
            if distance < - self.tol:
                error_map_x = np.dot(ebar_x_T, error_scaling_mat.T)
                jac = np.dot(error_map_x, jac)
            else:
                jac = np.zeros_like(x)
        return distance, jac
