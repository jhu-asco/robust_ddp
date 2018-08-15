#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import argparse
from optimal_control_framework.dynamics import CasadiUnicycleDynamics
from optimal_control_framework.costs import LQRObstacleCost
from optimal_control_framework.mpc_solvers import Ddp
from robust_ddp.robust_ddp import RobustDdp
from robust_ddp.robust_lqr_obstacle_cost import RobustLQRObstacleCost
from robust_ddp.obstacle_with_buffer import BufferedSphericalObstacle
from ellipsoid_package.ellipsoid_projection import plotEllipse
from ellipsoid_package.ellipsoid_helper import findEllipsoid
from optimal_control_framework.discrete_integrators import EulerIntegrator
from optimal_control_framework.sampling import DiscreteSampleTrajectories
from matplotlib.patches import Circle as CirclePatch

def createObstacleList(obs_params):
    obs_list = []
    for i, param in enumerate(obs_params):
        radius = max(param[2], 0.1)
        obs_list.append(BufferedSphericalObstacle(param[:2], radius))
    return obs_list

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
ts = np.arange(N + 1) * dt
kSigma = 1
# Obstacles
max_iters = 30
max_ko = 5000
ko_gain = 2.0/(0.8*max_iters)
# Desired terminal condition
ud = np.array([5.0, 0.0])
max_step = 5.0  # Allowed step for control

def regularDdp():
    us0 = np.tile(ud, (N, 1))
    BufferedSphericalObstacle.buffer_ellipsoid = None
    BufferedSphericalObstacle.buf = 0.05
    cost = LQRObstacleCost(N, Q, R, Qf, xd, ko=0, obstacles=obs_list, ud=ud)
    ddp = Ddp(dynamics, cost, us0, x0, dt, max_step, integrator=integrator)
    V = ddp.V
    print("Regular_V0: ", V)
    for i in range(max_iters):
        cost.ko = np.tanh(i*ko_gain)*max_ko
        ddp.update_dynamics(ddp.us, ddp.xs)
        ddp.iterate()
        V = ddp.V
    # Reset buffer
    BufferedSphericalObstacle.buf = 0.0
    print("Regular_Vfinal: ", V)
    return ddp.us, ddp.Ks

def singleTrial(M=100, plot=False, sigma_inflation=1.0):
    # Run regular ddp first:
    us_opt, Ks_opt = regularDdp()
    BufferedSphericalObstacle.sigma_inflation = sigma_inflation
    # ksigma is cost on trying to minimize ellipsoids
    cost = RobustLQRObstacleCost(N, Q, R, Qf, xd, ko=5000, obstacles=obs_list,
                                 kSigma = kSigma, ud=ud)
    ddp = RobustDdp(dynamics, cost, us_opt, x0, dt, max_step, Sigma0, Sigma_w,
                    integrator=integrator, Ks=Ks_opt)
    V = ddp.V
    print("V0: ", V)
    for i in range(max_iters):
        #cost.ko = np.tanh(i*ko_gain+0.5*max_iters)*max_ko
        #ddp.update_dynamics(ddp.us, ddp.xs)
        ddp.iterate()
        V = ddp.V
    print("Vfinal: ", V)
    # Sample example trajectories
    # Stdeviation!!:
    Sigma0_sqr = np.square(Sigma0)
    Sigmaw_sqr = np.square(Sigma_w)
    ws_sampling_fun = lambda : np.random.multivariate_normal(np.zeros(dynamics.n),
                                                             Sigmaw_sqr)
    x0_sampling_fun = lambda : np.random.multivariate_normal(x0, Sigma0_sqr)
    sampler = DiscreteSampleTrajectories(dynamics, integrator, cost,
                                         ws_sampling_fun, x0_sampling_fun)
    cost.ko = 0  #Ignore obstacle avoidance when computing costs
    BufferedSphericalObstacle.buffer_ellipsoid = None
    cost.ksigma = 0
    xss, uss, Jss = sampler.sample(M, ts, ddp)
    collision_array = np.full(M, False)
    for i, sample_traj in enumerate(xss):
        collision_array[i] = sampler.isColliding(obs_list, sample_traj)
    Ncollisions = np.sum(collision_array)
    print("Ncollisions: ", Ncollisions)
    print("Jmean: ", np.mean(Jss[:, 0]))
    print("Jstd: ", np.std(Jss[:, 0]))
    sigma_infl_str = '{0:.1f}'.format(sigma_inflation)
    sigma_infl_str = sigma_infl_str.replace('.','_')
    if plot:
        f = plt.figure(1)
        plt.clf()
        ax = f.add_subplot(111)
        ax.set_aspect('equal')
        plt.plot(ddp.xs[:, 0], ddp.xs[:, 1], 'b*-')
        for j in range(M):
            if collision_array[j]:
                color='m*-'
            else:
                color='g*-'
            plt.plot(xss[j][:,0], xss[j][:, 1], color)
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
        ax.set_xlim(left=-Sigma0[0,0])
        ax.set_ylim(bottom=-Sigma0[1,1])
        plt.tight_layout()
        plt.savefig('trajectories_'+sigma_infl_str+'.eps', bbox_inches='tight')
        plt.figure(2)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(ts[:-1], ddp.us[:, 0])
        plt.ylabel('Velocity (m/s)')
        plt.subplot(2, 1, 2)
        plt.plot(ts[:-1], ddp.us[:, 1])
        plt.ylabel('Angular rate (rad/s)')
        plt.tight_layout()
        plt.savefig('controls_'+sigma_infl_str+'.eps', bbox_inches='tight')
        plt.figure(3)
        plt.clf()
        plt.plot(ts, ddp.xs[:, 2])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (radians)')
        plt.tight_layout()
        plt.savefig('angle_'+sigma_infl_str+'.eps', bbox_inches='tight')
    return Ncollisions, Jss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test robust ddp')
    parser.add_argument('--pickle_file', type=str)
    parser.add_argument('--sigma_inflation', type=float, default=2.0)
    args = parser.parse_args()
    pickle_args = pickle.load(open(args.pickle_file, 'rb'))
    J0 = pickle_args['J0']
    M = pickle_args['M']
    Mtrials = pickle_args['Mtrials']
    obs_params = pickle_args['obs_params']
    xds = pickle_args['xds']
    delta_Js_frame = pickle_args['delta_Js_frame']
    mu0 = pickle_args['mu0']
    Sigma0 = pickle_args['sigma0']
    mud = pickle_args['mud']
    Sigmad = pickle_args['sigmad']
    Sigma_w = pickle_args['sigmaw']
    buf_array = pickle_args['buf_array']
    obs_list = createObstacleList(pickle_args['obs_mu'])
    Sigma0[2] = 1e-2  # TODO Change unicycle obst avoidance in opt framework
    Sigma0 = np.diag(Sigma0)
    Sigma_w = np.diag(Sigma_w)
    xd = mud
    x0 = mu0
    Ncollisions, Jss = singleTrial(M, True, args.sigma_inflation)
    Jss_list = [Jss[:,0]]
    Mtotal = M
    for trial_number in range(Mtrials):
        print("Trial number: ", trial_number)
        obs_list = createObstacleList(obs_params[trial_number])
        xd = xds[trial_number]
        Ncol_trial, Js_trial = singleTrial(M, False, args.sigma_inflation)
        Jss_list.append(Js_trial[:,0])
        Ncollisions = Ncollisions + Ncol_trial
        Mtotal = Mtotal + M
    Ncollisions = float(Ncollisions)/Mtotal*1000
    delta_Js_frame['gaussian_buffer'] = np.hstack(Jss_list) - J0
    plt.figure(4)
    ax = sns.barplot(data=delta_Js_frame)
    ax.set_xlabel('Buffer Length(m)')
    ax.set_ylabel('Change Trajectory Cost')
    plt.tight_layout()
    plt.savefig('cost_vs_buf_length.eps', bbox_inches='tight')
    plt.figure(5)
    plt.semilogy(buf_array, pickle_args['Ncollisions_buf'], 'b*-')
    plt.plot([buf_array[0], buf_array[-1]],
             [Ncollisions, Ncollisions], 'r-')
    plt.legend(['Obstacle Buffer', 'Variable Gaussian Buffer'])
    plt.xlabel('Buffer Length(m)')
    plt.ylabel('Ncollisions per 1000 samples')
    plt.tight_layout()
    plt.savefig('ncollisions_vs_buf_length.eps', bbox_inches='tight')
    plt.show()
    
