import math

import numpy as np

""" A controller class which implements a joint feedback plus feedforward
controller using a PID control law on the joint position error, desired acceleration
feedforward, and inertia and gravity compensation."""


class Control:

    def __init__(self, arm):
        # Bind arm
        self.arm = arm

        # Control type
        self.control_type = 'joint'

        # Joint space trajectory
        self.trajectory_available = False
        self.js_trajectory_q = None
        self.js_trajectory_dq = None
        self.js_trajectory_ddq = None

        # External force function
        self.external_force_available = False
        self.fext_function = None  # Not used in this control

        # Control parameters
        self.kp = 10
        self.kd = math.sqrt(self.kp)
        self.ki = 0
        self.eint = 0
        self.qprev = self.arm.q

        # Desired states
        self.q_des = np.zeros((self.arm.DOF, 1))
        self.dq_des = np.zeros((self.arm.DOF, 1))
        self.ddq_des = np.zeros((self.arm.DOF, 1))

    def set_joint_space_target(self, q_des):
        ''' Sets a joint position target directly'''
        assert np.shape(q_des) == (self.arm.DOF, 1)
        self.q_des = q_des

    def set_joint_space_trajectory(self, q_des, dq_des, ddq_des):
        ''' Sets a joint trajectory which the Simulator class will iterate during
        simulation and update the desired states in function of time'''
        assert np.shape(q_des)[0] == self.arm.DOF + 1
        self.trajectory_available = True
        self.js_trajectory_q = q_des
        self.js_trajectory_dq = dq_des
        self.js_trajectory_ddq = ddq_des

    def control(self):
        ''' Implements the control law'''

        # Get state
        q = self.arm.q
        dq = self.arm.dq

        # Error
        e_ = self.q_des - q
        edot = self.dq_des - dq
        self.eint = self.eint + e_ * self.arm.dt

        # Required pid accelerations
        acc_p = self.kp * e_
        acc_d = self.kd * edot
        acc_i = self.ki * self.eint
        acc_fb = acc_p + acc_i + acc_d

        # Feedforward torque
        acc_ff = self.ddq_des

        # Total torque
        acc = acc_fb + acc_ff

        # Compensation terms
        inert = self.arm.inertia()
        grav = self.arm.gravity()

        # Compute desired torque with inertia and gravity compensation
        tau = np.dot(inert, np.resize(acc, (self.arm.DOF, 1))) + grav

        return e_, acc_p, acc_i, acc_d, tau
