from math import *

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def plot_frame_t(t, ax, text=''):
    axis_length = 0.05
    r = t[0:3, 0:3]
    x = t[0][3]
    y = t[1][3]
    z = t[2][3]
    pose_ix = np.dot(r, np.array([axis_length, 0, 0]))
    pose_iy = np.dot(r, np.array([0, axis_length, 0]))
    pose_iz = np.dot(r, np.array([0, 0, axis_length]))
    ax.plot(x + [0, pose_ix[0]], y + [0, pose_ix[1]], z + [0, pose_ix[2]], 'r', linewidth=2)
    ax.plot(x + [0, pose_iy[0]], y + [0, pose_iy[1]], z + [0, pose_iy[2]], 'g', linewidth=2)
    ax.plot(x + [0, pose_iz[0]], y + [0, pose_iz[1]], z + [0, pose_iz[2]], 'b', linewidth=2)
    pose_t = np.dot(r, np.array([0.3 * axis_length, 0.3 * axis_length, 0.3 * axis_length]))
    ax.text(x + pose_t[0], y + pose_t[1], z + pose_t[2], text, fontsize=11)


class FiveDofArm:
    
    def __init__(self):
        
        # Init
        self.num_joints = 5
        self.num_links = 6
        
        # Function dictionaries
        self._Tx = {}  # for transform calculations
        self._T_inv = {}  # for inverse transform calculations
        self._J = {}  # for Jacobian calculations
        self._M = []  # placeholder for (x,y,z) inertia matrices
        self._Mq = None  # placeholder for joint space inertia matrix function
        self._Mq_g = None  # placeholder for joint space gravity term function
        
        # Set up our joint angle symbols
        self.q = [sp.Symbol('q%i' % ii) for ii in range(self.num_joints)]
        self.dq = [sp.Symbol('dq%i' % ii) for ii in range(self.num_joints)]
        self.x = [sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z')]
        
        # Gravity in cartesian space
        self.gravity = sp.Matrix([[0, 0, -9.81, 0, 0, 0]]).T
        
        # Inertia matrices for each link
        self._M.append(np.diag([1.0, 1.0, 1.0, 0.02, 0.02, 0.02]))  # link0
        self._M.append(np.diag([2.5, 2.5, 2.5, 0.04, 0.04, 0.04]))  # link1
        self._M.append(np.diag([5.7, 5.7, 5.7, 0.06, 0.06, 0.04]))  # link2
        self._M.append(np.diag([3.9, 3.9, 3.9, 0.055, 0.055, 0.04]))  # link3
        self._M.append(np.copy(self._M[1]))  # link4
        self._M.append(np.copy(self._M[1]))  # link5
        self._M.append(np.diag([0.7, 0.7, 0.7, 0.01, 0.01, 0.01]))  # link6
    
    def forward_kinematics(self, q, name):
        
        # Transformation matrices
        T_01 = np.mat([[cos(q[0]), 0, sin(q[0]), 0.01352 * cos(q[0])],
                       [sin(q[0]), 0, -cos(q[0]), 0.01352 * sin(q[0])],
                       [0, 1, 0, 0.09745],
                       [0, 0, 0, 1]])
        T_12 = np.mat([[cos(q[1] + pi / 2), -sin(q[1] + pi / 2), 0, 0.12 * cos(q[1] + pi / 2)],
                       [sin(q[1] + pi / 2), cos(q[1] + pi / 2), 0, 0.12 * sin(q[1] + pi / 2)],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        T_23 = np.mat([[cos(q[2] + pi / 2), 0, sin(q[2] + pi / 2), 0],
                       [sin(q[2] + pi / 2), 0, -cos(q[2] + pi / 2), 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])
        T_34 = np.mat([[cos(q[3]), 0, -sin(q[3]), 0],
                       [sin(q[3]), 0, cos(q[3]), 0],
                       [0, -1, 0, 0.12104],
                       [0, 0, 0, 1]])
        T_45 = np.mat([[cos(q[4] - pi / 2), -sin(q[4] - pi / 2), 0, 0.124 * cos(q[4] - pi / 2)],
                       [sin(q[4] - pi / 2), cos(q[4] - pi / 2), 0, 0.124 * sin(q[4] - pi / 2)],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        T_5EE = np.mat([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])
        
        if name == 'joint1':
            T = T_01
        elif name == 'joint2':
            T = T_01 * T_12
        elif name == 'joint3':
            T = T_01 * T_12 * T_23
        elif name == 'joint4':
            T = T_01 * T_12 * T_23 * T_34
        elif name == 'joint5':
            T = T_01 * T_12 * T_23 * T_34 * T_45
        elif name == 'EE':
            T = T_01 * T_12 * T_23 * T_34 * T_45 * T_5EE
        else:
            raise Exception('Invalid transformation name: %s' % name)
        return T


if __name__ == "__main__":
    # Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', xlim=(-0.2, 0.2), ylim=(-0.2, 0.2))
    
    # Init pos
    q_init = np.zeros((5, 1))
    # q_init[0] = pi/8
    # q_init[1] = -pi/8
    # q_init[2] = pi/4
    # q_init[3] = pi/8
    # q_init[4] = pi/8
    
    # Create arm
    arm = FiveDofArm()
    
    print(np.array(arm.forward_kinematics(q_init, 'EE')))
