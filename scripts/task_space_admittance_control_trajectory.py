from src.Interpolator import *
from src.Simulator import *
from src.arms import *
from src.controllers.task_space_admittance_controller import *

""" This script uses joint impedance control to track a trajectory under an external force."""

# Create robot
q_init = np.array([[math.pi / 2], [0.0]])
robot = TwoDofArm()
robot.set_q_init(q_init)

# Create controller
controller = Control(robot)

# Create task space trajectory (x and y)
p1 = np.resize(robot.forward_kinematics(q_init), (2,))
p2 = np.array([1.0, 1.0])
k_traj = 100  # Needs to be equal to the amount of animation frames = 100 (Set in Simulator class)
t = 5  # Needs to be equal to the simulation time of 100 frames
x_des, dx_des, ddx_des, dddx_des = interpolate_cubic(p1, p2, k_traj, t)
controller.set_task_space_trajectory(x_des, dx_des, ddx_des)

# Create external force in task space (20N in positive x-direction)
frames = 100  # Amount of animation iterations before restart
force_x = np.repeat(20, frames)
force_x[0:20] = 0  # Start force at frame 20
force_x[50:-1] = 0  # End force at frame 50
force_y = np.repeat(0, frames)
force = np.vstack((force_x, force_y))
controller.set_external_force_function(force)

# Run animation
joint_of_interest = 1
sim = Simulator(robot, controller, joint_of_interest)
sim.simulate()
