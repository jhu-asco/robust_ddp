#!/usr/bin/env python3
from optimal_control_framework.mpc_solvers import Ddp
from optimal_control_framework.costs import SphericalObstacle
from linearized_ellipsoid_propagation import propagateEllipsoid
import numpy as np

class RobustDdp(Ddp):
    def __init__(self, dynamics, cost, us0, x0, dt, max_step, Sigma0=0,
                 Sigmaw=0, use_prev_x=False):
        self.Sigmaw =Sigmaw
        self.Sigma = np.empty((cost.N+1, dynamics.n, dynamics.n))
        self.Sigma[0] = Sigma0
        super(RobustDdp, self).__init__(dynamics, cost, us0, x0, dt, max_step,
                                        use_prev_x=use_prev_x)

    def update_dynamics(self, us, xs):
        self.V = 0
        K = np.zeros((self.dynamics.m, self.dynamics.n))
        SphericalObstacle.updatePreviousX(None)
        for i, u in enumerate(us):
            x = xs[i]
            self.V = self.V + self.cost.stagewise_cost(i, x, u,
                                                       False, self.Sigma[i])
            xdot = self.dynamics.xdot(i, x, u, self.w)
            xs[i+1] = x + self.dt*xdot  # For now euler integration
            SphericalObstacle.updatePreviousX(x)
            # TODO Change instead of dynamics take in an integrator that
            # integrates continuous dynamics using a fancy integrator maybe
            jac = self.getDiscreteJacobians(i, x, u)
            Sigma = propagateEllipsoid(self.Sigma[i], self.Sigmaw, jac, K)
            self.Sigma[i+1] = Sigma
        self.V = self.V + self.cost.terminal_cost(xs[-1], False,
                                                  Sigma)
        SphericalObstacle.updatePreviousX(None)

    def getDiscreteJacobians(self, k, x, u):
        A, B, G = self.dynamics.jacobian(k, x, u, self.w)
        A = np.eye(self.dynamics.n) + A*self.dt
        B = B*self.dt
        G = G*self.dt
        return A, B, G

    def forward_pass_step(self, Ks, xs, us, alpha):
        Vnew = 0
        SphericalObstacle.updatePreviousX(None)
        for k in range(self.N):
            x = self.xs_up[k]
            delta_x = x - self.xs[k]
            K_k = Ks[k]
            u = self.us[k] + alpha*K_k[:, -1] + np.dot(K_k[:, :-1], delta_x)
            jac = self.getDiscreteJacobians(k, x, u)
            xdot = self.dynamics.xdot(k, x, u, self.w)
            Vnew = Vnew + self.cost.stagewise_cost(k, x, u, False,
                                                   self.Sigma[k])
            Sigma = propagateEllipsoid(self.Sigma[k], self.Sigmaw, jac,
                                       K_k[:, :-1])
            self.xs_up[k+1] = x + self.dt*xdot  # For now euler integration
            self.us_up[k] = u
            self.Sigma[k+1] = Sigma
            SphericalObstacle.updatePreviousX(x)
        Vnew = Vnew + self.cost.terminal_cost(self.xs_up[-1], False, Sigma)
        SphericalObstacle.updatePreviousX(None)
        return Vnew

