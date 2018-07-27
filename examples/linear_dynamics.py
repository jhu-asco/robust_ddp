#!/usr/bin/env python

from optimal_control_framework.dynamics import LinearDynamics
from robust_ddp.robust_ddp import RobustDdp
from robust_ddp.robust_lqr_obstacle_cost import RobustLQRObstacleCost
from ellipsoid_package.ellipsoid_projection import plotEllipse
from ellipsoid_package.ellipsoid_helper import findEllipsoid
import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(precision=3, suppress=True)
n = 2
m = 1
A = np.zeros((2,2))
A[0, 1] = 1
B = np.array([[0], [1]])
dynamics = LinearDynamics([A, B])
# Covariance
Sigma0 = np.diag([0.2, 0.2])
Sigma_w = 0*np.eye(n)
# Trajectory info
dt = 0.05
N = 40
#Q = np.zeros(dynamics.n)
#Q[1] = 0*dt  # For velocity along the trajectory
R = 0.01*dt*np.eye(dynamics.m)
Qf = 50*np.eye(dynamics.n)
Q = Qf*dt*0.1
ts = np.arange(N+1)*dt
xd = np.array([1, 0])
cost = RobustLQRObstacleCost(N, Q, R, Qf, xd, kSigma=0*dt)
max_step = 50.0  # Allowed step for control

x0 = np.array([0, 0])
us0 = np.zeros([N, dynamics.m])
ddp = RobustDdp(dynamics, cost, us0, x0, dt, max_step, Sigma0, Sigma_w)
V = ddp.V
for i in range(50):
    ddp.iterate()
    V = ddp.V
    print("V: ", V)
    print("xn: ", ddp.xs[-1])
    #print("Sigman: ", ddp.Sigma[-1])
f = plt.figure(1)
ax = f.add_subplot(111)
ax.plot(ddp.xs[:, 0], ddp.xs[:, 1])
ax.plot(xd[0], xd[1], 'r*')
ax.set_xlabel('x (m)')
ax.set_ylabel('velocity (m/s)')
for i, Sigma_i in enumerate(ddp.Sigma):
  ellipse = findEllipsoid(ddp.xs[i], Sigma_i)
  plotEllipse(n, ellipse, ax)
plt.figure(2)
plt.plot(ts[:-1], ddp.us[:, 0])
plt.ylabel('Acceleration (m/ss)')
plt.show()
