#!/usr/bin/env python

from optimal_control_framework.dynamics import CasadiUnicycleDynamics
from optimal_control_framework.integrators import EulerIntegrator
from robust_ddp.robust_ddp import RobustDdp
from robust_ddp.robust_lqr_obstacle_cost import RobustLQRObstacleCost
from robust_ddp.obstacle_with_buffer import BufferedSphericalObstacle
from ellipsoid_package.ellipsoid_projection import plotEllipse
from ellipsoid_package.ellipsoid_helper import findEllipsoid
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch
import seaborn as sns
import os

np.random.seed(1000)
sns.set_style('whitegrid')
sns.set(font_scale=1.0)
np.set_printoptions(precision=3, suppress=True)

dynamics = CasadiUnicycleDynamics()
integrator = EulerIntegrator(dynamics)
# Trajectory info
dt = 0.1
N = 20
Q = dt*np.zeros(dynamics.n)
R = 5*dt*np.eye(dynamics.m)
Qf = 30*np.eye(dynamics.n)
Qf[-1, -1] = 0
ts = np.arange(N+1)*dt
# Obstacles
kSigma = 1
max_iters = 40
max_ko = 10000
ko_gain = 2.0/(0.8*max_iters)
sigma_inflation = 3
obs1 = BufferedSphericalObstacle(np.array([3, 2]), 1)
obs2 = BufferedSphericalObstacle(np.array([6, 5]), 1)
obs_list = [obs1, obs2]
# Covariance:
Sigma0 = np.diag([0.05, 0.05, 0.0])
Sigma_w = 2*np.diag([0.01, 0.01, 0.01])
# Desired terminal condition
xd = np.array([8.0, 8.0, 0.0])
ud = np.array([5.0, 0.0])
cost = RobustLQRObstacleCost(N, Q, R, Qf, xd, ko=0, obstacles=obs_list,
                             kSigma = 1, ud=ud)
max_step = 1  # Allowed step for control

x0 = np.array([0, 0, 0])
us0 = np.tile(ud, (N, 1))
BufferedSphericalObstacle.sigma_inflation = sigma_inflation
ddp = RobustDdp(dynamics, cost, us0, x0, dt, max_step, Sigma0, Sigma_w)
V = ddp.V
for i in range(max_iters):
    cost.ko = np.tanh(i*ko_gain)*max_ko
    ddp.update_dynamics(ddp.us, ddp.xs)
    ddp.iterate()
    V = ddp.V
    print("V: ", V)
    print("xn: ", ddp.xs[-1])
f = plt.figure(1)
#plt.clf()
ax = f.add_subplot(111)
ax.set_aspect('equal')
plt.plot(ddp.xs[:, 0], ddp.xs[:, 1], 'b*-')
plt.plot(xd[0], xd[1], 'r*')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
for obs in obs_list:
  circ_patch = CirclePatch(obs.center, obs.radius, fill=False,
                           ec='r')
  ax.add_patch(circ_patch)
  ax.plot(obs.center[0], obs.center[1], 'r*')
for i, Sigma_i in enumerate(ddp.Sigma):
  ellipse = findEllipsoid(ddp.xs[i], sigma_inflation*Sigma_i)
  plotEllipse(3, ellipse, ax)
ax.set_xlim(left=-sigma_inflation*Sigma0[0,0])
ax.set_ylim(bottom=-sigma_inflation*Sigma0[1,1])
try:
    os.makedires('./results/unicycle')
except:
    pass
plt.savefig('./results/unicycle/unicycle_trajectory.eps', bbox_inches='tight')
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(ts[:-1], ddp.us[:, 0])
plt.ylabel('Velocity (m/s)')
plt.subplot(2,1,2)
plt.plot(ts[:-1], ddp.us[:, 1])
plt.ylabel('Angular rate (rad/s)')
plt.savefig('./results/unicycle/unicycle_controls.eps', bbox_inches='tight')
plt.figure(3)
plt.plot(ts, ddp.xs[:,2])
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (radians)')
plt.savefig('./results/unicycle/unicycle_residual_states.eps', bbox_inches='tight')
plt.show()
