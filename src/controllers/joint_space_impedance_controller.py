import math

import numpy as np

""" A controller class which implements a joint space impedance controller where joint torques are sensed and result
in an impedance behavior of the joint"""


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
        self.fext_function = None
        
        # Control parameters
        self.kp = 10
        self.kd = math.sqrt(25)
        
        # Impedance parameters
        self.imp_m = 10
        self.imp_k = (1 / math.radians(10))
        self.imp_b = 0.1
        
        # Desired states
        self.q_des = np.zeros((self.arm.DOF, 1))
        self.dq_des = np.zeros((self.arm.DOF, 1))
        self.ddq_des = np.zeros((self.arm.DOF, 1))
        
        # External force
        self.fext = None
    
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
    
    def set_external_force_function(self, f):
        ''' Sets a joint force function which the Simulator class will iterate during
        simulation and update the desired force in function of time'''
        assert np.shape(f)[0] == self.arm.DOF
        self.fext_function = f
    
    def control(self):
        ''' Implements the control law'''
        
        # Error
        e_ = self.q_des - self.arm.q
        edot = self.dq_des - self.arm.dq
        
        # Required pid torques
        tau_p = self.kp * e_
        tau_d = self.kd * edot
        tau_fb = tau_p + tau_d
        
        # Generate the mass matrix in joint space
        Mq = self.arm.inertia()
        
        # Required impedance joint torques
        desired_impedance_acc = (1 / self.imp_m) * (self.fext - self.imp_b * self.arm.dq - self.imp_k * e_)
        desired_impedance_torque = np.dot(Mq, desired_impedance_acc)
        
        # Gravitation compensation term
        grav = self.arm.gravity()
        
        # Total torque
        tau = tau_fb + desired_impedance_torque + grav
        
        return e_, tau_p, tau_d, np.zeros((self.arm.DOF, 1)), tau
