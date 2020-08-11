import numpy as np
import sympy as sp

'''This script calculates the position Jacobian for general open chains using the symbolic package 'sympy' '''


def calc_transform(q, l):
    Torg0 = sp.Matrix([[sp.cos(q[0]), -sp.sin(q[0]), 0, 0, ],
                       [sp.sin(q[0]), sp.cos(q[0]), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    
    T01 = sp.Matrix([[1, 0, 0, 0],
                     [0, sp.cos(q[1]), -sp.sin(q[1]), l[0] * sp.cos(q[1])],
                     [0, sp.sin(q[1]), sp.cos(q[1]), l[0] * sp.sin(q[1])],
                     [0, 0, 0, 1]])
    
    T12 = sp.Matrix([[1, 0, 0, 0],
                     [0, sp.cos(q[2]), -sp.sin(q[2]), 0],
                     [0, sp.sin(q[2]), sp.cos(q[2]), 0],
                     [0, 0, 0, 1]])
    
    T23 = sp.Matrix([[sp.cos(q[3]), 0, sp.sin(q[3]), 0],
                     [0, 1, 0, l[1]],
                     [-sp.sin(q[3]), 0, sp.cos(q[3]), 0],
                     [0, 0, 0, 1]])
    
    T34 = sp.Matrix([[1, 0, 0, 0],
                     [0, sp.cos(q[4]), -sp.sin(q[4]), 0],
                     [0, sp.sin(q[4]), sp.cos(q[4]), 0],
                     [0, 0, 0, 1]])
    
    T45 = sp.Matrix([[sp.cos(q[5]), 0, sp.sin(q[5]), 0],
                     [0, 1, 0, l[2]],
                     [-sp.sin(q[5]), 0, sp.cos(q[5]), 0],
                     [0, 0, 0, 1]])
    
    # Compute total transformation matrix
    T = Torg0 * T01 * T12 * T23 * T34 * T45
    
    return T


def calc_jacobian(q, l):
    # Create empty Jacobian of only the positions (not rotations)
    J = np.zeros((3, 6))
    
    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[1])
    s1 = np.sin(q[1])
    c3 = np.cos(q[3])
    s3 = np.sin(q[3])
    c4 = np.cos(q[4])
    s4 = np.sin(q[4])
    
    c12 = np.cos(q[1] + q[2])
    s12 = np.sin(q[1] + q[2])
    
    l1 = l[0]
    l3 = l[1]
    l5 = l[2]
    
    # Note that these equations come from the calculated derivatives below in the script
    J[0, 0] = -l1 * c0 * c1 - l3 * c0 * c12 - l5 * ((s0 * s3 - s12 * c0 * c3) * s4 + c0 * c4 * c12)
    J[1, 0] = -l1 * s0 * c1 - l3 * s0 * c12 + l5 * ((s0 * s12 * c3 + s3 * c0) * s4 - s0 * c4 * c12)
    J[2, 0] = 0
    
    J[0, 1] = (l1 * s1 + l3 * s12 + l5 * (s4 * c3 * c12 + s12 * c4)) * s0
    J[1, 1] = -(l1 * s1 + l3 * s12 + l5 * s4 * c3 * c12 + l5 * s12 * c4) * c0
    J[2, 1] = l1 * c1 + l3 * c12 - l5 * (s4 * s12 * c3 - c4 * c12)
    
    J[0, 2] = (l3 * s12 + l5 * (s4 * c3 * c12 + s12 * c4)) * s0
    J[1, 2] = -(l3 * s12 + l5 * s4 * c3 * c12 + l5 * s12 * c4) * c0
    J[2, 2] = l3 * c12 - l5 * (s4 * s12 * c3 - c4 * c12)
    
    J[0, 3] = -l5 * (s0 * s3 * s12 - c0 * c3) * s4
    J[1, 3] = l5 * (s0 * c3 + s3 * s12 * c0) * s4
    J[2, 3] = -l5 * s3 * s4 * c12
    
    J[0, 4] = l5 * ((s0 * s12 * c3 + s3 * c0) * c4 + s0 * s4 * c12)
    J[1, 4] = l5 * ((s0 * s3 - s12 * c0 * c3) * c4 - s4 * c0 * c12)
    J[2, 4] = -l5 * (s4 * s12 - c3 * c4 * c12)
    
    J[0, 5] = 0
    J[1, 5] = 0
    J[2, 5] = 0
    
    return J


if __name__ == "__main__":
    
    # Set up our joint angle symbols (6th angle doesn't affect any kinematics)
    dofs = 6
    q = [sp.Symbol('q0'), sp.Symbol('q1'), sp.Symbol('q2'), sp.Symbol('q3'),
         sp.Symbol('q4'), sp.Symbol('q5')]
    
    # Set up our arm segment length symbols
    l = [sp.Symbol('l1'), sp.Symbol('l3'), sp.Symbol('l5')]
    
    # Compute symbolic transformation matrix
    T = calc_transform(q, l)
    print("\nTotal symbolic transformation matrix from base to TCP: \n" + str(T))
    
    # Position of the TCP in end-effector frame (origin of end-effector frame)
    x = sp.Matrix([0, 0, 0, 1])
    
    # Compute symbolic forward kinematics
    Tx = T * x
    print("\nX-coordinate: " + str(Tx[0]))
    print("Y-coordinate: " + str(Tx[1]))
    print("Z-coordinate: " + str(Tx[2]))
    
    # Compute Jacobian elements (derivatives of each Cartesian coordinate to each joint variable)
    for ii in range(dofs):
        print("\nDerivative to joint: " + str(q[ii]))
        print("\tX-derivative: " + str(sp.simplify(Tx[0].diff(q[ii]))))
        print("\tY-derivative: " + str(sp.simplify(Tx[1].diff(q[ii]))))
        print("\tZ-derivative: " + str(sp.simplify(Tx[2].diff(q[ii]))))
    
    # Compute Jacobian with the computed derivatives and substituted values
    q0 = [0, 0, 0, 0, 0, 0]
    l0 = [1, 2, 3]
    J = calc_jacobian(q0, l0)
    print("\nJacobian matrix: \n" + str(J))
