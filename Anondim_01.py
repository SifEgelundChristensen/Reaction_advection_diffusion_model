# numerical solution system iteration 1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# length in meter
L = 21
# constant diffusion coefficient
Da = 0.5
alpha = 0.5


# number of steps in x direction
Nx =100
# step length
Dx = 1/(Nx-1)

# time steps (chosen based on stability criteria for explicit scheme)
dt = 0.4 * Dx**2  
# number of time steps in days t=100
Nt = int(30/ dt)

# decay constant
theta = (L**2*alpha)/Da
k = -theta

#spacial grid
x = np.linspace(0,1,Nx)



# initial condition
g=15
A = (np.exp(-g*x)-np.exp(-g))/(1-np.exp(-g))
A_IC = A.copy()
# boundary conditions
A[0] = 1.0
A[-1] = 0.0



# Set up the figure and axis
fig, ax = plt.subplots()
line, = ax.plot(x, A)
ax.set_xlabel('gamma')
ax.set_ylabel('a')
ax.set_title('Numerical solution to a along with analytical steady state')
ax.set_ylim(0, np.max(A)*1.1)

# Text annotation for timestep
time_text = ax.text(0.7, 0.9, '', transform=ax.transAxes, fontsize=12, color='darkred')
error_text = ax.text(0.6, 0.85, '', transform=ax.transAxes, fontsize=12, color='darkred')


## analytical solution for comparison
lambd = np.sqrt(theta)
steady_state = np.zeros_like(x)
steady_state[0] = 1 
steady_state[-1] = 0

steady_state[1:-1] = (np.sinh(lambd*(1-x[1:-1])))/(np.sinh(lambd))


# Evaluate and plot the analytical steady state on the same axes
ana_line, = ax.plot(x, steady_state, 'k--', lw=2, label='analytical steady state')
ax.legend()
ax.set_ylim(0, max(np.max(A), np.max(steady_state)) * 1.1)

#ana_lineIC, = ax.plot(x, A_IC, 'k--', lw=2, label='IC')
ax.legend()

def update(frame):
    global A
    A_new = A.copy()
    for i in range(1, Nx-1):
        # Finite difference for diffusion (explicit scheme)
        A_new[i] = (A[i] + dt * ((A[i+1] - 2*A[i] + A[i-1]) / Dx**2 + k * A[i]))
    # Apply boundary conditions
    A_new[0] = 1    # A(0, t) = 1
    A_new[-1] = 0   # A(L, t) = 0
    A = A_new
    line.set_ydata(A)

    # analytical steady-state is time-independent, keep plotted
    ana_line.set_ydata(steady_state)

    # IC
    #ana_lineIC.set_ydata(A_IC)

    error = np.mean(A - steady_state)


    # Update timestep display
    time_text.set_text(f"t = {frame * dt:.2f}")
    error_text.set_text(f"avg error = {error:.10f}")
   
    return line, ana_line, time_text, error_text #, ana_lineIC

# Create animation
ani = FuncAnimation(fig, update, frames=range(0, Nt, 10), interval=10, blit=False, repeat=False)
plt.rcParams['animation.embed_limit'] = 2**128
plt.show()
ani.save("diffusion_animation.gif", writer="pillow", fps=20)
