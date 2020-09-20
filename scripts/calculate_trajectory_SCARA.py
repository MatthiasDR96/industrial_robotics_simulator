import math

import matplotlib.pyplot as plt
import numpy as np


def rot_trans_to_matrix(rot, trans):
    mat = np.hstack((rot, trans))
    mat = np.vstack((mat, [0, 0, 0, 1]))
    return mat


def euler_to_rotation(r, p, y):
    Rr = np.mat([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Rp = np.mat([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Ry = np.mat([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return np.array(Rr * Rp * Ry)


def plot_frame_t(t, ax, text=''):
    axis_length = 0.05
    r = t[0:3, 0:3]
    x = t[0][3]
    y = t[1][3]
    z = t[2][3]
    pose_ix = np.dot(r, np.array([axis_length, 0, 0]))
    pose_iy = np.dot(r, np.array([0, axis_length, 0]))
    pose_iz = np.dot(r, np.array([0, 0, axis_length]))
    ax.plot(x + [0, pose_ix[0]], y + [0, pose_ix[1]], z + [0, pose_ix[2]], 'r', linewidth=2)
    ax.plot(x + [0, pose_iy[0]], y + [0, pose_iy[1]], z + [0, pose_iy[2]], 'g', linewidth=2)
    ax.plot(x + [0, pose_iz[0]], y + [0, pose_iz[1]], z + [0, pose_iz[2]], 'b', linewidth=2)
    pose_t = np.dot(r, np.array([0.3 * axis_length, 0.3 * axis_length, 0.3 * axis_length]))
    ax.text(x + pose_t[0], y + pose_t[1], z + pose_t[2], text, fontsize=11)


def quaternion_to_euler(qw, qx, qy, qz):
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return -roll, -pitch, -yaw


if __name__ == '__main__':

    # Params
    T = 2  # Period of the movement
    f = 100  # Sample frequency sent to the controller
    k_traj = f * T  # Amount of samples
    T0 = np.array([[-1, 0, 0, -0.2], [0, 1, 0, 0.25], [0, 0, -1, 0.2], [0, 0, 0, 1]])
    T1 = np.array([[0, 1, 0, 0], [1, 0, 0, 0.4], [0, 0, -1, 0.1], [0, 0, 0, 1]])

    # Time axis
    t = np.linspace(0, T, k_traj + 1)

    # Position interpolation
    X = np.array([[-0.2], [0.25], [0.2]]) + ((3 / T ** 2) * t ** 2 - (2 / T ** 3) * t ** 3) * np.array(
        [[0.2], [0.15], [-0.1]])
    Xdot = ((6 / T ** 2) * t - (6 / T ** 3) * t ** 2) * np.array([[0.2], [0.15], [-0.1]])
    Xdotdot = ((6 / T ** 2) - (12 / T ** 3) * t) * np.array([[0.2], [0.15], [-0.1]])

    # Plot trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0], X[1], X[2])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Position trajectory')

    # Plot Position
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(t, X[0], label='x')
    plt.plot(t, X[1], label='y')
    plt.plot(t, X[2], label='z')
    plt.title("Position")
    plt.xlabel('t (s)')
    plt.ylabel('x, y, and z (m)')
    plt.legend()

    # Plot Velocity
    plt.subplot(1, 3, 2)
    plt.plot(t, Xdot[0], label='x')
    plt.plot(t, Xdot[1], label='y')
    plt.plot(t, Xdot[2], label='z')
    plt.title("Velocity")
    plt.xlabel('t (s)')
    plt.ylabel('xdot, ydot, and zdot (m/s)')
    plt.legend()

    # Plot Acceleration
    plt.subplot(1, 3, 3)
    plt.plot(t, Xdotdot[0], label='x')
    plt.plot(t, Xdotdot[1], label='y')
    plt.plot(t, Xdotdot[2], label='z')
    plt.title("Acceleration")
    plt.xlabel('t (s)')
    plt.ylabel('xdotdot, ydotdot, and zdotdot (m/s^2)')
    plt.legend()

    # Quaternion interpolation
    q0 = np.array([0, 0, 1, 0])
    q1 = np.array([0, np.sqrt(0.5), np.sqrt(0.5), 0])
    omega = np.radians(45)
    so = np.sin(omega)
    st = (3 / T ** 2) * t ** 2 - (2 / T ** 3) * t ** 3
    quaternion_w = np.sin((1 - st) * omega) / so * q0[0] + np.sin(st * omega) / so * q1[0]
    quaternion_x = np.sin((1 - st) * omega) / so * q0[1] + np.sin(st * omega) / so * q1[1]
    quaternion_y = np.sin((1 - st) * omega) / so * q0[2] + np.sin(st * omega) / so * q1[2]
    quaternion_z = np.sin((1 - st) * omega) / so * q0[3] + np.sin(st * omega) / so * q1[3]

    # Convert to euler angles
    roll = []
    pitch = []
    yaw = []
    for i in range(len(t)):
        roll_, pitch_, yaw_ = quaternion_to_euler(quaternion_w[i], quaternion_x[i], quaternion_y[i], quaternion_z[i])
        roll.append(roll_)
        pitch.append(pitch_)
        yaw.append(yaw_)

    # Plot orientation quaternions
    plt.figure()
    plt.plot(t, quaternion_x, label='qx')
    plt.plot(t, quaternion_y, label='qy')
    plt.plot(t, quaternion_z, label='qz')
    plt.plot(t, quaternion_w, label='qw')
    plt.title("Quaternion interpolation")
    plt.xlabel('t (s)')
    plt.ylabel('qx, qy, qz, and qw')
    plt.legend()

    # Plot orientation euler angles
    plt.figure()
    plt.plot(t, roll, label='roll')
    plt.plot(t, pitch, label='pitch')
    plt.plot(t, yaw, label='yaw')
    plt.title("Euler angles")
    plt.xlabel('t (s)')
    plt.ylabel('roll, pitch, and yaw')
    plt.legend()

    # Plot transformation matrices
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', xlim=(-0.3, 0.1), ylim=(0.0, 0.5), zlim=(0.0, 0.3))
    ax.set_xlabel("X")
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_frame_t(np.identity(4), ax)
    transformations = []
    for i in range(len(t)):
        trans = [[X[0][i]], [X[1][i]], [X[2][i]]]
        rot = euler_to_rotation(roll[i], pitch[i], yaw[i])
        mat = rot_trans_to_matrix(rot, trans)
        transformations.append(mat)
        if i == 0 or i == k_traj / 2 or i == k_traj:
            plot_frame_t(mat, ax)
    plt.show()

    print("\nTransformation matrix at the beginning of the trajectory: \n" + str(transformations[0]))
    print("\nTransformation matrix at the end of the trajectory: \n" + str(transformations[-1]))
    print("\nTransformation matrix in the middle of the trajectory: \n" + str(transformations[int(k_traj / 2)]))
