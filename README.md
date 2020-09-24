# Industrial Robotics Simulator

This is a Python package for simple robot control as an addition to the lectures
of Industrial Robotics at the KU Leuven Campus in Bruges.

## Installation

It is sufficient to just download Python (3.6) and to download the package as a .zip folder from github.
Once the package is opened in your favorite IDE (I recommend Pycharm), you're ready to go.

## Simulator Structure

### Src

In the src folder, all the robot models and controllers are situated as well as a class for
linear interpolation and the main simulator class.

#### Simulator class

The Simulator class is the main class of the simulator and contains all functions for rendering the simulator.
The function accepts a robot model and a controller. The simulator is set to run for 100 frames at 0.05
frames per second. In the animate function, there is iterated over possible trajectories or external force
functions and the control law is executed. Finally the model is stepped one forward and the results are plotted.

#### Interpolator class

The interpolator class implements functions for linear interpolation using polynomial or trapezoidal
time scaling functions. A function accepts a starting point (with n degrees of freedom),
and end point, an amount of samples k_traj, and a traveling time t. Each function returns a list of
positions, velocities, accelerations and jerks in function of time.

#### Controllers

In the controller folder, all the controllers are implemented:

-   joint space feedback controller
-   joint space feedforward controller
-   joint space feedback plus feedforward controller
-   joint space impedance controller
-   task space force controller
-   task space impedance controller
-   task space motion controller

Each controller has some functions to set a target, trajectory, or external force and implements a function 'control'
which outputs the error, PID torques (if available), and the total output torque which is send to the robot.

#### Arms

In the arms folder, all robot models are implemented:

-   One degree of freedom link
-   Two degree of freedom link
-   Three degree of freedom link
-   UR5 robot model

Only the one, two, and three dof links can be used in the simulation. The UR5 model class just shows how a general
robot model including kinematics and dynamics can be set up calculating Jacobians, inertia, and gravity matrices.
For complex robots, the Coriolis matrix is not calculated due to complexity. Errors in the control due to this
mismodeling are compensated by feedback terms.

### Scripts

The scripts folder contains all the examples of using different robot models and controllers.
When you run each script you should be able to see the simulation run. You can adapt the robot model,
start position, trajectory, and external forces. Pay attention when changing variables. When receiving an
'assertion error', you probably entered e.g. a position with 2 dofs wile working with a 3 dof robot.

## Errata

It is definitely possible that bugs are present in the code or that some theory pieces are wrongly implemented.
If you notice any of these, please contact me.


