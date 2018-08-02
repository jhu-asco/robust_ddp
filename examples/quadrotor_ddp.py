#!/usr/bin/env python

from optimal_control_framework.dynamics import CasadiQuadrotorDynamics
from optimal_control_framework.discrete_integrators import (
    SemiImplicitQuadIntegrator)

from robust_ddp.robust_ddp import RobustDdp
from robust_ddp.robust_lqr_obstacle_cost import RobustLQRObstacleCost
from robust_ddp.obstacle_with_buffer import BufferedSphericalObstacle

from ellipsoid_package import EllipsoidTool
from ellipsoid_package.ellipsoid_projection import set_axes_equal, plotEllipsoid3D
from ellipsoid_package.ellipsoid_helper import findEllipsoid

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_style('whitegrid')
sns.set(font_scale=1.2)
np.set_printoptions(precision=3, suppress=True)

dynamics = CasadiQuadrotorDynamics(g=[0,0,-10], mass=1.5)
integrator = SemiImplicitQuadIntegrator(dynamics)
#integrator = None
# Trajectory info
dt = 0.02
N = 100
Q = dt*np.zeros(dynamics.n)
R = 0.1*dt*np.eye(dynamics.m)
#R[0,0] = 1e-3  # Thrust
R[0, 0] = 1*dt
Qf_arr = np.zeros(dynamics.n)
Qf_arr[:3] = 100  # Position
Qf_arr[3:6] = 100  # Velocity
Qf_arr[6:9] = 10  # RPY
Qf_arr[9:12] = 1  # 0mega
Qf = np.diag(Qf_arr)
ts = np.arange(N+1)*dt
# Obstacles
ko = 1000
obs1 = BufferedSphericalObstacle(np.array([0.6, 0.6, 0.9]), 0.2)
obs2 = BufferedSphericalObstacle(np.array([0.3, 0.3, 0.3]), 0.2)
obs_list = [obs1, obs2]
# Covariance
Sigma0 = 0.001*np.eye(dynamics.n)
Sigma0[:3, :3] = 0.1*np.eye(3)
Sigma_w = 0.1*np.eye(dynamics.n)  # For now
# Cost
# Add ud with default thrust as gravity maybe
xd = np.zeros(dynamics.n)
xd[:3] = 1.0
ud = np.zeros(dynamics.m)
ud[0] = 10 # Negative of gravity
cost = RobustLQRObstacleCost(N, Q, R, Qf, xd, ko=ko, obstacles=obs_list, ud=ud)
max_step = 100.0  # Allowed step for control

x0 = np.zeros(dynamics.n)
us0 = np.tile(ud, (N, 1))
ddp = RobustDdp(dynamics, cost, us0, x0, dt, max_step, Sigma0, Sigma_w,
                integrator=integrator)
V = ddp.V
for i in range(50):
    ddp.iterate()
    V = ddp.V
    print("V: ", V)
    print("xn_pos: ", ddp.xs[-1][:3])
    if not ddp.status:
        break
# %%
f = plt.figure(1)
plt.clf()
ax = f.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.plot(ddp.xs[:, 0], ddp.xs[:, 1], ddp.xs[:, 2])
ax.plot([xd[0]], [xd[1]], [xd[2]], 'r*')
ellipsoid_tool = EllipsoidTool()
for obs in obs_list:
    ellipsoid_tool.plotEllipsoid(obs.center, [obs.radius]*3, np.eye(3),
                                 ax=ax, cageColor='r')
sub_sample = 5
for i, Sigma_i in enumerate(ddp.Sigma):
  if i % sub_sample == 0:
    ellipse = findEllipsoid(ddp.xs[i], Sigma_i)
    print("Ellipse: ", ellipse.radii)
    plotEllipsoid3D(dynamics.n, ellipse, ax)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
set_axes_equal(ax)

plt.figure(2)
plt.subplot(2,2,1)
plt.plot(ts[:-1], ddp.us[:, 0])
plt.ylabel('Thrust (N)')
body_axes = ['x', 'y', 'z']
for i in range(3):
    plt.subplot(2,2,i+2)
    plt.plot(ts[:-1], ddp.us[:, i+1])
    plt.ylabel('Torque '+body_axes[i]+'(Nm)')
# Plot all states later
plt.show()
