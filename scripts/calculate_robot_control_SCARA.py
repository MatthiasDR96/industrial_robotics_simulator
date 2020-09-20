import matplotlib.pyplot as plt
import numpy as np

# Control params
Kp = 2
Ki = 2

# Solution values
e_0 = np.array([[-1.0], [0.46], [-0.1], [-0.526]])
tau = 1 / Kp
t = np.linspace(0, 2, 100)

# Solution
e_t = np.exp(- t / tau) * e_0

# Plot
plt.figure()
plt.plot(t, e_t[0], label='e_q0')
plt.plot(t, e_t[1], label='e_q1')
plt.plot(t, e_t[2], label='e_q2')
plt.plot(t, e_t[3], label='e_q3')
plt.plot(t, np.repeat(0, 100), 'r')
plt.xlabel('t (s)')
plt.ylabel('e(t) (rad)')
plt.title('Kp = ' + str(Kp))
plt.legend()
plt.show()
