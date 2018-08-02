from optimal_control_framework.costs import LQRObstacleCost
from obstacle_with_buffer import BufferedSphericalObstacle
import numpy as np

class RobustLQRObstacleCost(LQRObstacleCost):
    def __init__(self, N, Q, R, Qf, xd=None, obstacles=[], ko = 1,
                 obstacle_class=BufferedSphericalObstacle, kSigma=1, ud=None):
        super(RobustLQRObstacleCost, self).__init__(
            N, Q, R, Qf, xd, obstacles, ko, ud)
        self.obstacle_class = obstacle_class
        self.kSigma = kSigma

    def findSigmaCost(self, sigma_radius, kSigma):
        return 0.5*np.sum(np.square(sigma_radius*kSigma))

    def addSigmaCost(self, out, sigma_cost, compute_grads):
        if not compute_grads:
            out = out + sigma_cost
        else:
            out[0] = out[0] + sigma_cost
        return out

    def stagewise_cost(self, i, x, u, compute_grads=False, Sigma=None):
        if Sigma is not None:
            self.obstacle_class.set_buffer_ellipsoid(Sigma)
            sigma_cost = self.findSigmaCost(
                self.obstacle_class.buffer_ellipsoid[1],
                self.kSigma)
        else:
            sigma_cost = 0
        out = super(RobustLQRObstacleCost, self).stagewise_cost(i, x, u,
                                                                compute_grads)
        out = self.addSigmaCost(out, sigma_cost, compute_grads)
        return out

    def terminal_cost(self, xf, compute_grads=False, Sigma=None):
        if Sigma is not None:
            self.obstacle_class.set_buffer_ellipsoid(Sigma)
            sigma_cost = self.findSigmaCost(
                self.obstacle_class.buffer_ellipsoid[1],
                self.kSigma)
        else:
            sigma_cost = 0
        out = super(RobustLQRObstacleCost, self).terminal_cost(
            xf, compute_grads)
        out = self.addSigmaCost(out, sigma_cost, compute_grads)
        return out 
