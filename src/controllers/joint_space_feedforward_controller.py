import math

import numpy as np

""" A controller class which implements a joint feedforward
controller by compensating for the desired acceleration torque and the desired gravity torque."""


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
        
        # Compensation terms
        inert = self.arm.inertia(self.q_des)
        grav = self.arm.gravity(self.q_des)
        
        # Compute desired torque with inertia and gravity compensation
        tau = np.dot(inert, self.ddq_des) + grav
        
        return np.zeros((self.arm.DOF, 1)), np.zeros((self.arm.DOF, 1)), np.zeros((self.arm.DOF, 1)), np.zeros(
            (self.arm.DOF, 1)), tau
