from industrial_robotics_simulator.Interpolator import *
from industrial_robotics_simulator.Simulator import *
from industrial_robotics_simulator.arms import *
from industrial_robotics_simulator.controllers.joint_space_admittance_controller import *

""" This script uses joint impedance control to track a trajectory under an external force."""

# Create robot
q_init = np.array([[3 * math.pi / 4], [0.0]])
robot = TwoDofArm()
robot.set_q_init(q_init)

# Create controller
controller = Control(robot)

# Create joint space trajectory
p1 = np.reshape(q_init, (robot.DOF,))
p2 = np.array([math.pi / 4, 0.0])
k_traj = 100  # Needs to be equal to the amount of animation frames = 100 (Set in Simulator class)
t = 5  # Needs to be equal to the simulation time of 100 frames
q_des, dq_des, ddq_des, dddq_des = interpolate_cubic(p1, p2, k_traj, t)
controller.set_joint_space_trajectory(q_des, dq_des, ddq_des)

# Create external force in joint space (20Nm about joint 1)
frames = 100  # Amount of animation iterations before restart
torque_joint1 = np.repeat(20, frames)
torque_joint1[0:70] = 0  # Start torque at frame 70
torque_joint2 = np.repeat(0, frames)
torque = np.vstack((torque_joint1, torque_joint2))
controller.set_external_force_function(torque)

# Run animation
joint_of_interest = 1
sim = Simulator(robot, controller, joint_of_interest)
sim.simulate()
