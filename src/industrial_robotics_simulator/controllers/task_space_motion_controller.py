import math

import numpy as np

""" A controller class which implements a feedback
controller in task space and inertia and gravity compensation.
A velocity component is substracted from the torque for stability"""


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
        self.fext_function = None  # Not used in this control

        # Control parameters
        self.kp = 25
        self.kv = math.sqrt(25)

        # Desired states
        self.x_des = np.zeros((self.arm.DOF, 1))
        self.dx_des = np.zeros((self.arm.DOF, 1))
        self.ddx_des = np.zeros((self.arm.DOF, 1))

    def set_task_space_target(self, x_des):
        ''' Sets a task position target directly'''
        assert np.shape(x_des) == (2, 1)
        self.x_des = x_des

    def set_task_space_trajectory(self, x_des, dx_des, ddx_des):
        ''' Sets a Cartesian trajectory which the Simulator class will iterate during
        simulation and update the desired states in function of time'''
        self.trajectory_available = True
        self.ts_trajectory_x = x_des
        self.ts_trajectory_dx = dx_des
        self.ts_trajectory_ddx = ddx_des

    def control(self):
        ''' Implements the control law'''

        # Get state
        x = np.resize(self.arm.forward_kinematics(self.arm.q), (2, 1))

        # Calculate desired end-effector acceleration
        e_ = self.x_des - x  # Error in task space position

        # Required P acceleration
        acc = self.kp * e_  # Desired acceleration

        # Generate the mass matrix in joint space and in end-effector space
        Mq = self.arm.inertia()
        Mx = self.arm.gen_Mx()

        # Calculate desired end-effector force (in x and y) with inverse task dynamics
        Fx = np.dot(Mx[:, 0:2], np.resize(acc, (2, 1)))

        # Calculate the Jacobian at the end-effector
        jac = self.arm.generate_jacobian_ee(self.arm.q)

        # Calculate desired joint space torques
        Fq = np.dot(jac.T, Fx)

        # Compensation terms
        grav = self.arm.gravity()

        # Velocity compensation
        vel_comp = np.reshape(np.dot(Mq, self.kv * self.arm.dq), (self.arm.DOF, 1))

        # Total torque with gravity and velocity compensation
        tau = Fq + grav - vel_comp

        return e_, np.zeros((self.arm.DOF, 1)), np.zeros((self.arm.DOF, 1)), np.zeros((self.arm.DOF, 1)), tau
