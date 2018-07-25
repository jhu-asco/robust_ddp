#!/usr/bin/env python3
from optimal_control_framework.mpc_solvers import Ddp
from obstacle_with_buffer import BufferedSphericalObstacle
from linearized_ellipsoid_propagation import propagateEllipsoid
import numpy as np

class RobustDdp(Ddp):
    def __init__(self, *args, **kwargs):
        super(RobustDdp, self).__init__(*args, **kwargs)
        n = self.dynamics.n
        self.Sigma = np.empty((self.N+1, n, n))
        self.Sigma[0] = np.zeros((self.dynamics.n, self.dynamics.n))
        self.Sigmaw = 0

    def setCovariance(self, Sigma0, Sigmaw):
        self.Sigma[0] = Sigma0
        self.Sigmaw = Sigmaw

    def getDiscreteJacobians(self, k, x, u):
        A, B, G = self.dynamics.jacobian(k, x, u, self.w)
        A = np.eye(self.dynamics.n) + A*self.dt
        B = B*self.dt
        G = G*self.dt
        return A, B, G

    def forward_pass_step(self, Ks, xs, us, alpha):
        Vnew = 0
        for k in range(self.N):
            x = self.xs_up[k]
            delta_x = x - self.xs[k]
            K_k = Ks[k]
            u = self.us[k] + alpha*K_k[:, -1] + np.dot(K_k[:, :-1], delta_x)
            jac = self.getDiscreteJacobians(k, x, u)
            Sigma = propagateEllipsoid(self.Sigma[k], self.Sigmaw, jac,
                                       K_k[:, :-1])
            xdot = self.dynamics.xdot(k, x, u, self.w)
            Vnew = Vnew + self.cost.stagewise_cost(k, x, u, False, Sigma)
            self.xs_up[k+1] = x + self.dt*xdot  # For now euler integration
            self.us_up[k] = u
            self.Sigma[k+1] = Sigma
        Vnew = Vnew + self.cost.terminal_cost(self.xs_up[-1], False, Sigma)
        return Vnew

