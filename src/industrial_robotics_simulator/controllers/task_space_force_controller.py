import numpy as np

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
        self.kp = 1
        self.ki = 0
        self.eint = 0
        self.k_damp = 25

        # Desired force
        self.f_des = np.zeros((6, 1))

        # Previously executed joint torques
        self.tau_prev = np.zeros((self.arm.DOF, 1))

    def set_force_target(self, f_des):
        ''' Sets a force target directly'''
        assert np.shape(f_des) == (6, 1)
        self.f_des = f_des

    def control(self):
        ''' Implements the control law'''

        # Get end-effector Jacobian
        J = self.arm.generate_jacobian_ee(self.arm.q)

        # Calculate force the robot is applying in task space
        f_tip = np.dot(np.linalg.pinv(J).T, self.tau_prev)

        # Wall simulated at x=1.2m, if robot didn't reach wall, there is no external force
        # , otherwise, external force equals applied force
        if np.resize(self.arm.forward_kinematics(self.arm.q), (2,))[0] >= 1.2:
            f_ext = -f_tip
        else:
            f_ext = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

        # Error
        e_ = self.f_des - f_ext
        self.eint = self.eint + e_ * self.arm.dt

        # Feedback torques in task space
        f_p = self.kp * e_
        f_i = self.ki * self.eint
        f_fb = f_p + f_i

        # Feedforward torque in task space
        f_ff = self.f_des

        # Velocity compensation
        ee_twist = np.dot(J, self.arm.dq)
        vel_comp = np.reshape(self.k_damp * ee_twist, (6, 1))

        # Total torque
        f = f_ff + f_fb - vel_comp

        # Take only x-direction
        f[1] = 0
        f[-1] = 0

        # Compensation terms
        grav = self.arm.gravity()

        # Compute desired torque with only gravity compensation
        tau = np.dot(J.T, f) + grav
        self.tau_prev = tau

        # Limit commanded torques
        tau = np.clip(tau, -self.arm.tau_limit, self.arm.tau_limit)

        return e_, f_p, f_i, np.zeros((6, 1)), tau
