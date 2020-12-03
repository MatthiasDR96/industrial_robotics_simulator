import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

'''This script calculates the forward kinematics for the SCARA robot '''


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    x_mean = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]
    y_mean = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]
    z_mean = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])


def plot_frame_t(t, ax, text=''):
    axis_length = 0.1
    r = t[0:3, 0:3]
    x = t[0][3]
    y = t[1][3]
    z = t[2][3]
    pose_ix = np.dot(r, np.array([axis_length, 0, 0]))
    pose_iy = np.dot(r, np.array([0, axis_length, 0]))
    pose_iz = np.dot(r, np.array([0, 0, axis_length]))
    ax.plot([x, x + pose_ix[0]], [y, y + pose_ix[1]], [z, z + pose_ix[2]], 'r', linewidth=2)
    ax.plot([x, x + pose_iy[0]], [y, y + pose_iy[1]], [z, z + pose_iy[2]], 'g', linewidth=2)
    ax.plot([x, x + pose_iz[0]], [y, y + pose_iz[1]], [z, z + pose_iz[2]], 'b', linewidth=2)
    pose_t = np.dot(r, np.array([0.3 * axis_length, 0.3 * axis_length, 0.3 * axis_length]))
    ax.text(x + pose_t[0], y + pose_t[1], z + pose_t[2], text, fontsize=11)


def plot(T_, q_, ax):

    # Plot world frame
    plot_frame_t(np.identity(4), ax, 'w')

    # Value to substitute
    subst = [(q[0], q_[0]), (q[1], q_[1]), (q[2], q_[2]), (q[3], q_[3])]

    # Plot joint frames and links
    for i in range(0, 4):
        print("Transformation matrix T" + str(i) + str(i+1) + ": " + str(T_[i].subs(subst)))
        plot_frame_t(np.array(T_[i].subs(subst)), ax, 'j' + str(i + 1))
    set_axes_equal(ax)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')


def calc_transform(q):
    T01 = sp.Matrix([[-sp.cos(q[0]), sp.sin(q[0]), 0, -0.2 * sp.cos(q[0])],
                     [-sp.sin(q[0]), -sp.cos(q[0]), 0, -0.2 * sp.sin(q[0])],
                     [0, 0, 1, 0.3],
                     [0, 0, 0, 1]])

    T12 = sp.Matrix([[sp.sin(q[1]), sp.cos(q[1]), 0, 0.25 * sp.sin(q[1])],
                     [-sp.cos(q[1]), sp.sin(q[1]), 0, -0.25 * sp.cos(q[1])],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    T23 = sp.Matrix([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, q[2] - 0.1],
                     [0, 0, 0, 1]])

    T34 = sp.Matrix([[-sp.sin(q[3]), sp.cos(q[3]), 0, 0],
                     [sp.cos(q[3]), sp.sin(q[3]), 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])

    T02 = T01 * T12
    T03 = T01 * T12 * T23
    T = T01 * T12 * T23 * T34

    return T01, T02, T03, T


if __name__ == "__main__":

    # Set up our joint angle symbols for symbolic computation
    q = [sp.Symbol('q1'), sp.Symbol('q2'), sp.Symbol('q3'), sp.Symbol('q4')]

    # Compute symbolic transformation matrices
    T01, T02, T03, T = calc_transform(q)
    print("\nTotal symbolic transformation matrix from base to TCP: \n" + str(T))

    # Position of the TCP in end-effector frame (origin of end-effector frame)
    x = sp.Matrix([0, 0, 0, 1])

    # Compute symbolic forward kinematics position
    Tx = T * x
    print("\nX-coordinate: " + str(sp.simplify(Tx[0])))
    print("Y-coordinate: " + str(sp.simplify(Tx[1])))
    print("Z-coordinate: " + str(sp.simplify(Tx[2])))

    # Compute forward kinematics at home position (substitute real values in symbolic representation)
    home = [0, 0, 0, 0]
    T_ = T.subs([(q[0], home[0]), (q[1], home[1]), (q[2], home[2]), (q[3], home[3])])
    x_ = Tx.subs([(q[0], home[0]), (q[1], home[1]), (q[2], home[2]), (q[3], home[3])])
    print("\nT = " + str(np.resize(T_, (4, 4))))
    print("\nX = " + str(np.resize(x_, (4,))))

    # Plot forward kinematics
    print()
    pos_1 = [0, 0, 0, 0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot([T01, T02, T03, T], pos_1, ax)
    plt.show()