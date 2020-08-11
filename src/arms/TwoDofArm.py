from math import *

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

""" A class to calculate all the kinematics and dynamics
    for a two link arm. Also stores the mass of each of the links."""


class TwoDofArm:
    
    def __init__(self, dt=1e-5, singularity_threshold=0.00025):
        
        # Arm params
        self.DOF = 2
        self.dt = dt
        self.singularity_threshold = singularity_threshold
        self.params = {"mass": [1., 1.], "damping": [0.1, 0.1], "stiffness": [0], "inertia": [1., 1.],
                       "length": [1., 1.],
                       "center_of_gravity": [0.5, 0.5], "gravity": 9.81}
        
        # Create mass matrices at COM for each link
        self.M1 = np.zeros((6, 6))
        self.M2 = np.zeros((6, 6))
        self.M1[0:3, 0:3] = np.eye(3) * self.params["mass"][0]
        self.M1[5, 5] = self.params["inertia"][0]
        self.M2[0:3, 0:3] = np.eye(3) * self.params["mass"][1]
        self.M2[5, 5] = self.params["inertia"][1]
        
        # Arm state
        self.q = np.array([[0.0], [0.0]])
        self.dq = np.array([[0.0], [0.0]])
        
        # Arm state
        self.tau_limit = 20  # Nm
        
        # Simulation time
        self.time_elapsed = 0.0
        
        # Rest angles (for null space control)
        self.rest_angles = np.array([[0.0], [0.0]])
    
    def __str__(self):
        string = "Two degree of freedom arm:\n"
        for i in range(self.DOF):
            string += "\tLink " + str(i) + ": \n\t\tMass: " + str(self.params["mass"][0]) \
                      + "\n\t\tLength: " + str(self.params["length"][0]) \
                      + "\n\t\tInertia: " + str(self.params["inertia"][0]) \
                      + "\n\t\tCOM: " + str(self.params["center_of_gravity"][0]) + "\n"
        return string
    
    def set_q_init(self, q_init):
        assert len(q_init) == self.DOF
        self.q = q_init
    
    def plot(self, q=None):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax1.grid()
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        line1, = ax1.plot([], [], 'o-', lw=2)
        if not q is None:
            self.q = q
        [x, y] = self.position()
        line1.set_data(x, y)
    
    def position(self, q=None):
        if q is None:
            q = self.q
        x1 = self.params["length"][0] * np.cos(q[0])
        y1 = self.params["length"][0] * np.sin(q[0])
        x2 = self.params["length"][1] * np.cos(q[0] + q[1])
        y2 = self.params["length"][1] * np.sin(q[0] + q[1])
        x = np.cumsum([0, x1, x2])
        y = np.cumsum([0, y1, y2])
        return x, y
    
    def reset(self, q, dq):
        assert len(q) == self.DOF
        assert len(dq) == self.DOF
        self.q = q
        self.dq = dq
        self.time_elapsed = 0.0
    
    def forward_kinematics(self, q):
        assert len(q) == self.DOF
        T_01 = np.resize([[cos(q[0]), -sin(q[0]), 0, cos(q[0]) * self.params["length"][0]],
                          [sin(q[0]), cos(q[0]), 0, sin(q[0]) * self.params["length"][0]],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], (4, 4))
        T_12 = [[cos(q[1]), -sin(q[1]), 0, cos(q[1]) * self.params["length"][1]],
                  [sin(q[1]), cos(q[1]), 0, sin(q[1]) * self.params["length"][1]],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
        T = np.dot(T_01, T_12)
        
        # Return x and y
        fk = np.dot(T, np.resize([0, 0, 0, 1], (4, 1)))
        return fk[0:2]
    
    def inverse_kinematics(self, xy, q_init=None):
        
        if q_init is None:
            q_init = self.q
        
        def distance_to_target(q, xy):
            fxy = self.forward_kinematics(q)
            return np.sqrt((fxy[0] - xy[0]) ** 2 + (fxy[1] - xy[1]) ** 2)
        
        return scipy.optimize.minimize(fun=distance_to_target, x0=q_init, args=([xy[0], xy[1]]))['x']
    
    def forward_dynamics(self, q, dq, tau_=None):
        assert len(q) == self.DOF
        assert len(dq) == self.DOF
        if tau_ is None:
            tau_ = [[0.0], [0.0]]
        assert np.shape(tau_) == (self.DOF, 1)
        inert = self.inertia()
        grav = self.gravity()
        cor = self.coriolis(q, dq)
        damp = self.damping(dq)
        ddq = np.dot(np.linalg.inv(inert), (tau_ - cor - grav - damp))
        return np.reshape(ddq, (self.DOF, 1))
    
    def inverse_dynamics(self, q, dq, ddq):
        inert = self.inertia()
        grav = self.gravity()
        cor = self.coriolis(q, dq)
        damp = self.damping(dq)
        tau_ = np.dot(inert, np.resize(ddq, (self.DOF, 1))) + cor + grav + damp
        return np.reshape(tau_, (self.DOF, 1))
    
    def inertia(self, q=None):
        if q is None:
            q = self.q
        jac1 = self.generate_jacobian_com1(q)
        jac2 = self.generate_jacobian_com2(q)
        Mq = np.dot(jac1.T, np.dot(self.M1, jac1)) + \
             np.dot(jac2.T, np.dot(self.M2, jac2))
        return Mq
    
    def gravity(self, q=None):
        if q is None:
            q = self.q
        jac1 = self.generate_jacobian_com1(q)
        jac2 = self.generate_jacobian_com2(q)
        gr = np.dot(jac1.T, [0, self.params['gravity'], 0, 0, 0, 0]) + \
             np.dot(jac2.T, [0, self.params['gravity'], 0, 0, 0, 0])
        return np.resize(gr, (self.DOF, 1))
    
    def damping(self, dq):
        return np.resize([[self.params['damping'][0] * dq[0]], [self.params['damping'][1] * dq[1]]], (self.DOF, 1))
    
    def coriolis(self, theta, dtheta):
        m2 = self.params["mass"][1]
        L1 = self.params["length"][0]
        L2 = self.params["length"][1]
        c_theta_1 = - m2 * L1 * L2 * sin(theta[1]) * (2 * dtheta[0] * dtheta[1] + dtheta[1] ** 2)
        c_theta_2 = m2 * L1 * L2 * (dtheta[0] ** 2) * sin(theta[1])
        return np.resize([c_theta_1, c_theta_2], (2, 1))
    
    def generate_jacobian_com1(self, q):
        jac = np.zeros((6, self.DOF))
        jac[0, 0] = self.params["center_of_gravity"][0] * -np.sin(q[0])
        jac[1, 0] = self.params["center_of_gravity"][0] * np.cos(q[0])
        jac[5, 0] = 1.0
        return jac
    
    def generate_jacobian_com2(self, q):
        q0 = q[0]
        q01 = q[0] + q[1]
        jac = np.zeros((6, self.DOF))
        jac[0, 1] = self.params["center_of_gravity"][1] * -np.sin(q01)
        jac[1, 1] = self.params["center_of_gravity"][1] * np.cos(q01)
        jac[5, 1] = 1.0
        jac[0, 0] = self.params["center_of_gravity"][0] * -np.sin(q0) + jac[0, 1]
        jac[1, 0] = self.params["center_of_gravity"][0] * np.cos(q0) + jac[1, 1]
        jac[5, 0] = 1.0
        return jac
    
    def generate_jacobian_ee(self, q):
        q0 = q[0]
        q01 = q[0] + q[1]
        jac = np.zeros((6, self.DOF))
        jac[0, 1] = self.params["center_of_gravity"][1] * -np.sin(q01)
        jac[1, 1] = self.params["center_of_gravity"][1] * np.cos(q01)
        jac[5, 1] = 1.0
        jac[0, 0] = self.params["center_of_gravity"][0] * -np.sin(q0) + jac[0, 1]
        jac[1, 0] = self.params["center_of_gravity"][0] * np.cos(q0) + jac[1, 1]
        jac[5, 0] = 1.0
        return jac
    
    def gen_Mx(self):
        
        # Get inertia matrix in joint space
        Mq = self.inertia()
        
        # Get ee jacobian
        jac = self.generate_jacobian_ee(self.q)
        
        # Get inverse operational inertia matrix
        Mx_inv = np.dot(jac, np.dot(np.linalg.inv(Mq), jac.T))
        
        # Handle singularities
        if abs(np.linalg.det(np.dot(jac, jac.T))) > .005 ** 2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            # In the case that the robot is entering near singularity
            u, s, v = np.linalg.svd(Mx_inv)
            for i in range(len(s)):
                if s[i] < self.singularity_threshold:
                    s[i] = 0
                else:
                    s[i] = 1.0 / float(s[i])
            Mx = np.dot(v.T, np.dot(np.diag(s), u.T))
        
        return Mx
    
    def step(self, tau_, delta_t):
        
        # Compute accelerations
        ddq = self.forward_dynamics(self.q, self.dq, tau_)
        
        # Compute velocity
        self.dq += ddq * delta_t
        
        # Compute position
        self.q += self.dq * delta_t
        
        # Proceed elapsed time
        self.time_elapsed += delta_t
