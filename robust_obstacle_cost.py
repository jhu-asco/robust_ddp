#!/usr/bin/env python

from optimal_control_framework.costs.obstacle import SphericalObstacle

class RobustSphericalObstacle(SphericalObstacle):
    buffer_ellipsoid = None  # Principal axes (R.T), semi_major_axes
    def __init__(self, center, radius):
        super(RobustSphericalObstacle, self).__init__(center, radius)

    @classmethod
    def set_buffer_ellipsoid(self, Sigma):
        w, v = np.linalg.eigh(Sigma)
        self.buffer_ellipsoid = (v, w)

    def distance(self, x, compute_grads=False):
        if self.buffer_ellipsoid is None:
            self.buffer_ellipsoid = (1, 0)
        principal_axes, semi_major_axes = self.buffer_ellipsoid
        scale = semi_major_axes + self.radius
        z = self.mapState(x)
        error = z - self.center
        error_scaling_mat = self.radius*(principal_axes/scale[:, np.newaxis])
        error_map = np.dot(error_scaling_mat, error)
        distance, jac = super(RobustSphericalObstacle, self).distance_substep(
            error_map, compute_grads)
        if compute_grads:
            if distance < - self.tol:
                error_map_x = np.dot(z_x.T, error_scaling_mat.T)
                jac = np.dot(error_map_x, jac)
            else:
                jac = np.zeros_like(x)
        return distance, jac
