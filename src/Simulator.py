import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class Simulator:
    
    def __init__(self, model, control=None, joint_of_interest=1):
        
        # Model
        self.model = model
        
        # Joint of interest for plotting
        if 0 < joint_of_interest <= self.model.DOF:
            self.joint_of_interest = joint_of_interest - 1  # Convert from number to array index starting from 0
        else:
            print("No valid joint of interest, the robot only has " + str(self.model.DOF) + " joints.")
            exit(0)
        
        # Control
        self.control = control
        
        # Animation params
        self.frames = 100
        self.delta_t = 1 / 20  # 20 fps
        self.model.dt = self.delta_t
        self.sim_time = self.frames * self.delta_t
        
        # Data lists
        self.time_axis = []  # Time axis
        self.state_list = []  # Joint value of one joint of interest
        self.error_list = []  # Error in joint value for one joint of interest
        self.tau_p_list = []  # Joint torques for the P signal
        self.tau_i_list = []  # Joint torques for the I signal
        self.tau_d_list = []  # Joint torques for the D signal
        self.tau_list = []  # Total joint torques
        
        # Set up figure and animation
        self.fig = plt.figure()
        
        # Plot axis
        ax1 = self.fig.add_subplot(221, aspect='equal', autoscale_on=False, xlim=(-4, 4), ylim=(-4, 4))
        self.line_plot, = ax1.plot([], [], 'o-', lw=2)
        self.line_time = ax1.text(0.02, 0.90, '', transform=ax1.transAxes)
        ax1.set_title("Plot")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.grid()
        
        # State axis
        ax2 = self.fig.add_subplot(223, aspect='equal', autoscale_on=False, xlim=(0, self.sim_time * 10), ylim=(-5, 5))
        self.line_state, = ax2.plot([], [], '-', lw=1)
        ax2.set_title("State")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Theta (rad)")
        ax2.grid()
        
        # Error axis
        ax3 = self.fig.add_subplot(224, aspect='equal', autoscale_on=False, xlim=(0, self.sim_time * 10), ylim=(-5, 5))
        self.line_error, = ax3.plot([], [], '-', lw=1)
        ax3.set_title("Control error")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Error (rad)")
        ax3.grid()
        
        # Control axis
        ax4 = self.fig.add_subplot(222, aspect='equal', autoscale_on=False, xlim=(0, self.sim_time * 10),
                                   ylim=(-20, 20))
        self.line_tau_p, = ax4.plot([], [], '-', lw=1, label="P")
        self.line_tau_i, = ax4.plot([], [], '-', lw=1, label="I")
        self.line_tau_d, = ax4.plot([], [], '-', lw=1, label="D")
        self.line_tau, = ax4.plot([], [], '-', lw=1, label="Total")
        ax4.legend()
        ax4.set_title("Control commands")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Torque (Nm)")
        ax4.grid()
        
        # Choose the interval based on dt and the time to animate one step
        t0 = time.time()
        self.model.step(None, 0)  # Modulate one simulation step
        t1 = time.time()
        self.interval = 1000 * self.delta_t - (t1 - t0)
    
    # Initialize animation
    def init(self):
        self.line_plot.set_data([], [])
        self.line_time.set_text('')
        self.line_state.set_data([], [])
        self.line_error.set_data([], [])
        self.line_tau_p.set_data([], [])
        self.line_tau_d.set_data([], [])
        self.line_tau_i.set_data([], [])
        self.line_tau.set_data([], [])
    
    # Animation step
    def animate(self, i):
        
        print("Frame " + str(i) + " at simulation time " + str(self.model.time_elapsed))
        
        # Compute control signal
        if self.control:
            
            # Set external force
            if not self.control.fext_function is None:
                self.control.fext = np.reshape(self.control.fext_function[:, i], (2, 1))
            
            # Get next point from trajectory
            if self.control.trajectory_available:
                if self.control.control_type == 'task':
                    self.control.x_des = np.reshape(self.control.ts_trajectory_x[0:-1, i], (self.model.DOF, 1))
                    self.control.dx_des = np.reshape(self.control.ts_trajectory_dx[0:-1, i], (self.model.DOF, 1))
                    self.control.ddx_des = np.reshape(self.control.ts_trajectory_ddx[0:-1, i], (self.model.DOF, 1))
                else:
                    self.control.q_des = np.reshape(self.control.js_trajectory_q[0:-1, i], (self.model.DOF, 1))
                    self.control.dq_des = np.reshape(self.control.js_trajectory_dq[0:-1, i], (self.model.DOF, 1))
                    self.control.ddq_des = np.reshape(self.control.js_trajectory_ddq[0:-1, i], (self.model.DOF, 1))
            
            # Execute control law
            error, tau_p, tau_i, tau_d, tau = self.control.control()
            
            # Get data of joint of interest
            error_ = error[self.joint_of_interest]
            tau_p_ = tau_p[self.joint_of_interest]
            tau_i_ = tau_i[self.joint_of_interest]
            tau_d_ = tau_d[self.joint_of_interest]
            tau_ = tau[self.joint_of_interest]
        
        else:
            tau = None
            error_, tau_p_, tau_i_, tau_d_, tau_ = 0, 0, 0, 0, 0
        
        # New model state
        self.model.step(tau, self.delta_t)
        new_state = self.model.position()
        time_elapsed = self.model.time_elapsed
        
        # Arm plots
        self.line_plot.set_data(new_state)
        self.line_time.set_text('time = %.1f' % time_elapsed)
        
        # Update plot data
        self.time_axis.append(self.model.time_elapsed)
        self.state_list.append(self.model.q[self.joint_of_interest][0])
        self.error_list.append(error_)
        self.tau_p_list.append(tau_p_)
        self.tau_i_list.append(tau_i_)
        self.tau_d_list.append(tau_d_)
        self.tau_list.append(tau_)
        
        # Plot data lists
        self.line_state.set_data(self.time_axis, self.state_list)
        self.line_error.set_data(self.time_axis, self.error_list)
        self.line_tau_p.set_data(self.time_axis, self.tau_p_list)
        self.line_tau_i.set_data(self.time_axis, self.tau_i_list)
        self.line_tau_d.set_data(self.time_axis, self.tau_d_list)
        self.line_tau.set_data(self.time_axis, self.tau_list)
    
    def simulate(self):
        _ = animation.FuncAnimation(self.fig, self.animate, interval=self.interval,
                                    frames=self.frames, init_func=self.init)
        plt.show()
