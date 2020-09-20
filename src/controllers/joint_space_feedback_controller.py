import math

import numpy as np

""" A controller class which implements a joint feedback
controller using a PID control law on the joint position error."""


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

        # Required pid torques
        tau_p = self.kp * e_
        tau_i = self.ki * self.eint
        tau_d = self.kd * edot

        # Total torque
        tau = np.reshape(tau_p + tau_i + tau_d, (self.arm.DOF, 1))

        return e_, tau_p, tau_i, tau_d, tau
