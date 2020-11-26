import math

import numpy as np
import scipy.optimize

""" A controller class which implements a task space force feedback and feedforward
controller with velocity compensation."""


class Control:

    def __init__(self, arm):
        # Bind arm
        self.arm = arm

        # Control type
        self.control_type = 'task'

        # Task space trajectory
        self.trajectory_available = False
        self.ts_trajectory_x = None
        self.ts_trajectory_dx = None
        self.ts_trajectory_ddx = None

        # External force function
        self.external_force_available = False
        self.fext_function = None

        # Control parameters
        self.kp = 10
        self.kv = math.sqrt(10)

        # Impedance parameters
        self.imp_m = 10
        self.imp_k = 5
        self.imp_b = math.sqrt(self.imp_k)

        # Desired states
        self.x_des = np.zeros((self.arm.DOF, 1))
        self.dx_des = np.zeros((self.arm.DOF, 1))
        self.ddx_des = np.zeros((self.arm.DOF, 1))

        # Desired external force
        self.fext = None

    def set_task_space_target(self, x_des):
        ''' Sets a task position target directly'''
        assert np.shape(x_des) == (2, 1)
        self.x_des = x_des

    def set_task_space_trajectory(self, x_des, dx_des, ddx_des):
        ''' Sets a Cartesian trajectory which the Simulator class will iterate during
        simulation and update the desired states in function of time'''
        assert np.shape(x_des)[0] == 3
        self.trajectory_available = True
        self.ts_trajectory_x = x_des
        self.ts_trajectory_dx = dx_des
        self.ts_trajectory_ddx = ddx_des

    def set_external_force_function(self, f):
        ''' Sets a Cartesian force function which the Simulator class will iterate during
                simulation and update the desired force in function of time'''
        assert np.shape(f)[0] == 2
        self.fext_function = f

    def control(self):
        ''' Implements the control law'''

        # Calculate the Jacobian at the end-effector
        jac = self.arm.generate_jacobian_ee(self.arm.q)

        # Get robot state
        x = np.resize(self.arm.forward_kinematics(self.arm.q), (2, 1))
        dx = np.dot(scipy.linalg.pinv(jac).T, self.arm.dq)[0:2]

        # Calculate displacement from initial state
        e_ = self.x_des - x
        des_p_acc = self.kp * e_  # Desired acceleration

        # Generate the mass matrix in joint space and in end-effector space
        Mx = self.arm.gen_Mx()

        # Calculate desired P-control end-effector wrench (in x and y)
        des_p_wrench = np.dot(Mx[:, 0:2], np.resize(des_p_acc, (2, 1)))

        # Calculate desired joint space force to reach target
        des_p_torque = np.dot(jac.T, des_p_wrench)

        # Gravity compensation term
        grav = self.arm.gravity()

        # Desired end-effector acceleration for impedance behavior
        des_ee_acc = (1 / self.imp_m) * (self.fext - self.imp_b * dx - self.imp_k * e_)

        # Desired end-effector wrench for impedance behavior
        des_ee_wrench = np.dot(Mx[:, 0:2], np.resize(des_ee_acc, (2, 1)))

        # Desired joint torques for impedance behavior
        des_tau = np.dot(jac.T, des_ee_wrench)

        # Total torque
        tau = des_p_torque + grav + des_tau

        return e_, np.zeros((self.arm.DOF, 1)), np.zeros((self.arm.DOF, 1)), np.zeros((self.arm.DOF, 1)), tau
