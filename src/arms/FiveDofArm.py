from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import sympy as sp


class FiveDofArm:
    """ A class to calculate all the transforms and Jacobians
    for a five dof arm. Also stores the mass of each of the links."""
    
    def __init__(self, dt=1e-5, singularity_threshold=0.005):
        
        # Init
        self.DOF = 5
        self.num_links = 6
        self.dt = dt
        self.singularity_threshold = singularity_threshold
        
        # Set up our joint angle symbols
        self.q_sym = [sp.Symbol('q%i' % ii) for ii in range(self.DOF)]
        
        # Create mass matrices at COM for each link
        self.M = []
        self.M.append(np.diag([1.0, 1.0, 1.0, 0.02, 0.02, 0.02]))  # link0
        self.M.append(np.diag([2.5, 2.5, 2.5, 0.04, 0.04, 0.04]))  # link1
        self.M.append(np.diag([5.7, 5.7, 5.7, 0.06, 0.06, 0.04]))  # link2
        self.M.append(np.diag([3.9, 3.9, 3.9, 0.055, 0.055, 0.04]))  # link3
        self.M.append(np.copy(self.M[1]))  # link4
        self.M.append(np.copy(self.M[1]))  # link5
        self.M.append(np.diag([0.7, 0.7, 0.7, 0.01, 0.01, 0.01]))  # link6
        
        # Segment lengths associated with each joint
        L = np.array([0.0935, 0.13453, 0.4251, 0.12, 0.3921, 0.0935, 0.0935, 0.0935])
        
        # transform matrix from joint 0 to joint 1 reference frame, link 1 reference frame is the same as joint 1
        self.T01 = sp.Matrix([[sp.cos(self.q_sym[0]), 0, sp.sin(self.q_sym[0]), 0.01352 * sp.cos(self.q_sym[0])],
                       [sp.sin(self.q_sym[0]), 0, -sp.cos(self.q_sym[0]), 0.01352 * sp.sin(self.q_sym[0])],
                       [0, 1, 0, 0.09745],
                       [0, 0, 0, 1]])
        
        # Transform matrix from joint 1 to joint 2 reference frame
        self.T12 = sp.Matrix([[sp.cos(self.q_sym[1] + pi / 2), -sp.sin(self.q_sym[1] + pi / 2), 0, 0.12 * sp.cos(self.q_sym[1] + pi / 2)],
                       [sp.sin(self.q_sym[1] + pi / 2), sp.cos(self.q_sym[1] + pi / 2), 0, 0.12 * sp.sin(self.q_sym[1] + pi / 2)],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        # Transform matrix from joint 2 to joint 3
        self.T23 = sp.Matrix([[sp.cos(self.q_sym[2] + pi / 2), 0, sp.sin(self.q_sym[2] + pi / 2), 0],
                       [sp.sin(self.q_sym[2] + pi / 2), 0, -sp.cos(self.q_sym[2] + pi / 2), 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])
        
        # Transform matrix from joint 3 to joint 4
        self.T34 = sp.Matrix([[sp.cos(self.q_sym[3]), 0, -sp.sin(self.q_sym[3]), 0],
                       [sp.sin(self.q_sym[3]), 0, sp.cos(self.q_sym[3]), 0],
                       [0, -1, 0, 0.12104],
                       [0, 0, 0, 1]])
        
        # Transform matrix from joint 4 to joint 5
        self.T45 = sp.Matrix([[sp.cos(self.q_sym[4] - pi / 2), -sp.sin(self.q_sym[4] - pi / 2), 0, 0.124 * sp.cos(self.q_sym[4] - pi / 2)],
                       [sp.sin(self.q_sym[4] - pi / 2), sp.cos(self.q_sym[4] - pi / 2), 0, 0.124 * sp.sin(self.q_sym[4] - pi / 2)],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        # Transform matrix from joint 5 to end-effector
        self.T5EE = sp.Matrix([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])
        
        # Orientation part of the Jacobian (compensating for angular velocity)
        kz = sp.Matrix([0, 0, 1])  # Screw vector in local joint frame
        self.J_orientation = [
            self.calculate_transformation_matrix('joint0')[:3, :3] * kz,  # joint 0 orientation
            self.calculate_transformation_matrix('joint1')[:3, :3] * kz,  # joint 1 orientation
            self.calculate_transformation_matrix('joint2')[:3, :3] * kz,  # joint 2 orientation
            self.calculate_transformation_matrix('joint3')[:3, :3] * kz,  # joint 3 orientation
            self.calculate_transformation_matrix('joint4')[:3, :3] * kz,  # joint 4 orientation
            self.calculate_transformation_matrix('joint5')[:3, :3] * kz]  # joint 5 orientation
        
        # Arm state
        self.q = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        self.dq = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        
        # Simulation time
        self.time_elapsed = 0.0
        
        # For the null space controller, keep arm near these angles
        self.rest_angles = np.array([None, np.pi / 4.0, -np.pi / 2.0, np.pi / 4.0, np.pi / 2.0, np.pi / 2.0])
    
    def __str__(self):
        string = "UR5 robot arm:\n"
        for i in range(self.DOF):
            string += "\tLink " + str(i) + ": \n\t\tMass: " + str(self.M[i][0]) \
                      + "\n\t\tInertia: " + str(self.M[i][-1])
        return string
    
    def set_q_init(self, q_init):
        assert len(q_init) == self.DOF
        self.q = q_init
    
    def plot(self, q=None):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-4, 4), ylim=(-4, 4))
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
        x, y, z = self.forward_kinematics(q)
        x = np.cumsum([0, x])
        y = np.cumsum([0, y])
        return x, y
    
    def reset(self, q, dq):
        assert len(q) == self.DOF
        assert len(dq) == self.DOF
        self.q = q
        self.dq = dq
        self.time_elapsed = 0.0
    
    def forward_kinematics(self, q):
        
        # Total symbolic transformation matrix
        T = self.calculate_transformation_matrix('EE')
        
        # Transform x into world coordinates
        Tx = T * sp.Matrix([0, 0, 0, 1])
        
        # Substitute q in symbolic values
        Tx = Tx.subs(self.q_sym[0], q[0])
        Tx = Tx.subs(self.q_sym[1], q[1])
        Tx = Tx.subs(self.q_sym[2], q[2])
        Tx = Tx.subs(self.q_sym[3], q[3])
        Tx = Tx.subs(self.q_sym[4], q[4])
        Tx = Tx.subs(self.q_sym[5], q[5])
        
        return np.array(Tx[0:-1]).flatten()
    
    def inverse_kinematics(self, xy, q_init=None):
        
        if q_init is None:
            q_init = self.q_sym
        
        def distance_to_target(q, xy):
            fxy = self.forward_kinematics(q)
            return np.sqrt((fxy[0] - xy[0]) ** 2 + (fxy[1] - xy[1]) ** 2)
        
        return scipy.optimize.minimize(fun=distance_to_target, x0=q_init, args=([xy[0], xy[1]]))['x']
    
    def forward_dynamics(self, q, dq, tau_=None):
        assert len(q) == self.DOF
        assert len(dq) == self.DOF
        if tau_ is None:
            tau_ = [[0.0], [0.0], [0.0], [0.0], [0.0]]
        assert np.shape(tau_) == (self.DOF, 1)
        inert = self.inertia(q)
        grav = self.gravity(q)
        ddtheta = np.dot(np.linalg.pinv(inert), (tau_ - grav))
        return np.array(ddtheta, dtype='float')
    
    def inverse_dynamics(self, q, dq, ddq):
        inert = self.inertia(q)
        grav = self.gravity(q)
        tau_ = np.dot(inert, np.resize(ddq, (self.DOF, 1))) + grav
        return np.reshape(tau_, (self.DOF, 1))
    
    def calculate_transformation_matrix(self, name):
        
        if name == 'joint0'or name == 'link0':
            T = sp.Matrix(np.identity(self.DOF))
        elif name == 'joint1' or name == 'link1':
            T = self.T01
        elif name == 'joint2'or name == 'link2':
            T = self.T01 * self.T12
        elif name == 'joint3'or name == 'link3':
            T = self.T01 * self.T12 * self.T23
        elif name == 'joint4' or name == 'link4':
            T = self.T01 * self.T12 * self.T23 * self.T34
        elif name == 'joint5' or name == 'link5':
            T = self.T01 * self.T12 * self.T23 * self.T34 * self.T45
        elif name == 'link6' or name == 'EE':
            T = self.T01 * self.T12 * self.T23 * self.T34 * self.T45 * self.T5EE
        else:
            raise Exception('Invalid transformation name: %s' % name)
        return T
    
    def J(self, name):
        """ Calculates the transform for a joint or link
        name string: name of the joint or link, or end-effector
        q np.array: joint angles
        """
        # Calculate total transformation matrix
        T = self.calculate_transformation_matrix(name)
        
        # Transform x into world coordinates
        Tx = T * sp.Matrix([0, 0, 0, 1])
        
        # Calculate derivative of (x,y,z) wrt to each joint
        J = []
        for ii in range(self.DOF):
            J.append([])
            J[ii].append(Tx[0].diff(self.q_sym[ii]))  # dx/dq[ii]
            J[ii].append(Tx[1].diff(self.q_sym[ii]))  # dy/dq[ii]
            J[ii].append(Tx[2].diff(self.q_sym[ii]))  # dz/dq[ii]
        
        end_point = name.strip('link').strip('joint')
        if end_point != 'EE':
            end_point = min(int(end_point) + 1, self.DOF)
            # add on the orientation information up to the last joint
            for ii in range(end_point):
                J[ii] = sp.Matrix(list(J[ii]) + list(self.J_orientation[ii]))
            # fill in the rest of the joints orientation info with 0
            for ii in range(end_point, self.DOF):
                J[ii] = J[ii] + [0, 0, 0]
        J = sp.Matrix(np.reshape(J, (6, 6))).T
        return J
    
    def inertia(self, q):
        """ Calculates the joint space inertia matrix for the ur5
                q np.array: joint angles
                """
        # Get the Jacobians for each link's COM
        J = [self.J('link%s' % ii) for ii in range(self.num_links)]
        
        # Transform each inertia matrix into joint space and sum together the effects of each arm segments' inertia
        Mq = sp.zeros(self.DOF)  # Initialize an empty 6 x 6 matrix
        for ii in range(self.num_links):
            Mq += J[ii].T * self.M[ii] * J[ii]  # Convert inertia from COM to base frame
        Mq = sp.Matrix(Mq)
        
        # Substitute q in symbolic values
        Mq = Mq.subs(self.q_sym[0], q[0])
        Mq = Mq.subs(self.q_sym[1], q[1])
        Mq = Mq.subs(self.q_sym[2], q[2])
        Mq = Mq.subs(self.q_sym[3], q[3])
        Mq = Mq.subs(self.q_sym[4], q[4])
        Mq = Mq.subs(self.q_sym[5], q[5])
        return np.reshape(np.array(Mq, dtype='float'), (6, 6))
    
    def gravity(self, q):
        """ Calculates the force of gravity in joint space for the ur5
        q np.array: joint angles
        """
        
        # Get the Jacobians for each link's COM
        J = [self.J('link%s' % ii) for ii in range(self.num_links)]
        
        # Transform each inertia matrix into joint space and sum together the effects of each arm segments' inertia
        Mq_g = sp.zeros(self.DOF, 1)
        for ii in range(self.DOF):
            Mq_g += J[ii].T * self.M[ii] * sp.Matrix([[0, 0, -9.81, 0, 0, 0]]).T
        Mq_g = sp.Matrix(Mq_g)
        
        Mq_g = Mq_g.subs(self.q_sym[0], q[0])
        Mq_g = Mq_g.subs(self.q_sym[1], q[1])
        Mq_g = Mq_g.subs(self.q_sym[2], q[2])
        Mq_g = Mq_g.subs(self.q_sym[3], q[3])
        Mq_g = Mq_g.subs(self.q_sym[4], q[4])
        Mq_g = Mq_g.subs(self.q_sym[5], q[5])
        return np.resize(Mq_g, (self.DOF, 1))
    
    def gen_Mx(self):
        
        # Get inertia in joint space
        Mq = self.inertia(self.q)
        
        # Get ee jacobian
        jac = self.J('EE')
        
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
        
        # Get new state
        self.time_elapsed += delta_t
