import scipy.optimize
from src.plot import *
import math
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''This script calculates the inverse kinematics in 6dofs for a Fanuc CR-7iA robot'''


def forward_kinematics(q):

    T01 = sp.Matrix([[-sp.cos(q[0]), 0, -sp.sin(q[0]), 0.050 * sp.cos(q[0])],
                     [-sp.sin(q[0]), 0, sp.cos(q[0]), 0.050 * sp.sin(q[0])],
                     [0, 1, 0, 0.457],
                     [0, 0, 0, 1]])

    T12 = sp.Matrix([[-sp.sin(q[1]), -sp.cos(q[1]), 0, -0.440 * sp.sin(q[1])],
                     [sp.cos(q[1]), -sp.sin(q[1]), 0, 0.440 * sp.cos(q[1])],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    T23 = sp.Matrix([[-sp.cos(q[2]), 0, -sp.sin(q[2]), 0.035 * sp.cos(q[2])],
                     [-sp.sin(q[2]), 0, sp.cos(q[2]), 0.035 * sp.sin(q[2])],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])

    T34 = sp.Matrix([[-sp.cos(q[3]), 0, -sp.sin(q[3]), 0],
                     [-sp.sin(q[3]), 0, sp.cos(q[3]), 0],
                     [0, 1, 0, 0.420],
                     [0, 0, 0, 1]])

    T45 = sp.Matrix([[-sp.cos(q[4]), 0, -sp.sin(q[4]), 0],
                     [-sp.sin(q[4]), 0, sp.cos(q[4]), 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])

    T56 = sp.Matrix([[sp.cos(q[5]), -sp.sin(q[5]), 0, 0],
                     [sp.sin(q[5]), sp.cos(q[5]), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    T02 = T01 * T12
    T03 = T01 * T12 * T23
    T04 = T01 * T12 * T23 * T34
    T05 = T01 * T12 * T23 * T34 * T45
    T = T01 * T12 * T23 * T34 * T45 * T56

    return [T01, T02, T03, T04, T05, T]


def inverse_kinematics(T_target, q_init):

    def distance_to_target(q, T_target):

        # Forward kinematics
        [_, _, _, _, _, T] = forward_kinematics(q)
        T = np.array(T).astype(np.float64)

        # Position error
        target = T_target[:3, -1]
        squared_distance_to_target = np.linalg.norm(T[:3, -1] - target)

        # Orientation error
        target_orientation = T_target[:3, :3]
        squared_distance_to_orientation = np.linalg.norm(T[:3, :3] - target_orientation)

        # Total error
        squared_distance = squared_distance_to_target + squared_distance_to_orientation
        return squared_distance

    return scipy.optimize.minimize(fun=distance_to_target, x0=q_init, args=T_target)['x']


def plot_robot(q_, ax):

    # Set up our joint angle symbols for symbolic computation
    q = [sp.Symbol('q1'), sp.Symbol('q2'), sp.Symbol('q3'), sp.Symbol('q4'), sp.Symbol('q5'), sp.Symbol('q6')]

    # Get robot kinematics
    T_ = forward_kinematics(q)

    # Plot world frame
    plot_frame_t(np.identity(4), ax, 'w')

    # Value to substitute
    subst = [(q[0], q_[0]), (q[1], q_[1]), (q[2], q_[2]), (q[3], q_[3]), (q[4], q_[4]), (q[5], q_[5])]

    # Plot joint frames and links
    for i in range(0, 5):
        plot_frame_t(np.array(T_[i].subs(subst)), ax, 'j' + str(i + 1))
    set_axes_equal(ax)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    return np.array(T_[-1].subs(subst))


if __name__ == "__main__":

    # Initial config
    q_init = [0, 0, 0, 0, 0, 0]

    # Define target position
    T_target = np.array([[-math.sqrt(2)/2, math.sqrt(2)/2, 0, 0.4],
                         [math.sqrt(2)/2, math.sqrt(2)/2, 0, -0.4],
                         [0, 0, -1, 0.4],
                         [0, 0, 0, 1]])

    # Plot robot and target
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-1.0, 0.5)
    ax.set_zlim3d(0.0, 1.0)
    plot_robot(q_init, ax)
    plot_frame_t(T_target, ax, 'T')

    # Calculate inverse kinematics
    q = inverse_kinematics(T_target, q_init)
    print("Joint values: " + str(q))

    # Plot robot and target
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-1.0, 0.5)
    ax.set_zlim3d(0.0, 1.0)
    plot_robot(q, ax)
    plot_frame_t(T_target, ax, 'T')
    plt.show()