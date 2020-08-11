from src.Interpolator import *
from src.Simulator import *
from src.arms import *
from src.controllers.task_space_motion_controller import *

""" This script uses task space motion control to track a trajectory."""

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

# Run animation
sim = Simulator(robot, controller)
sim.simulate()
