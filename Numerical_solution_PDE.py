# numerical solution to B 1st iteration using steady state from A

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# length in meter
Lg = 1 # non-dimensional length is 1
Ls = 21 # length of original system for non-dimensionalization
# constant diffusion coefficient
Da = 0.5
Db = 0.5
alpha = 0.3
U = 0.1  # velocity




# D
D = Db/Da
# Theta
Theta = (alpha*Ls**2)/Da

# Peclet number (constant advection)
Pea=(U*Ls)
print("Peclet number:", Pea)
print("Theta:", Theta)

# number of steps in x direction
Nx =100
# step length
Dx = Lg/(Nx-1)


# time steps (chosen based on stability criteria for explicit scheme)
dt_diff = 0.5 * Dx**2 / (2*D)
# time steps to ensure stability for advection term
dt_adv = 0.5 * Dx / U
# choosing smallest of the two to ensure stability for both numerical schemes
dt = min(dt_diff, dt_adv)

# number of time steps in days t=100
Nt = int(300/ dt)


#spacial grid
x = np.linspace(0,Lg,Nx)



# initial condition and BC's
B =np.zeros_like(x)



## analytical solution at steady state for A, source term of equation for B
lambd = np.sqrt(Theta)
steady_stateA = np.zeros_like(x)
steady_stateA[0] = 1 
steady_stateA[-1] = 0

steady_stateA[1:-1] = (np.sinh(lambd*(1-x[1:-1])))/(np.sinh(lambd))


## analytical steady state for B
B0 = np.zeros_like(x)
A = -((Theta*np.exp(np.sqrt(Theta)))/(2*np.sinh(np.sqrt(Theta))))/(Theta-Pea*np.sqrt(Theta))
A2 = ((Theta*np.exp(-np.sqrt(Theta)))/(2*np.sinh(np.sqrt(Theta))))/(Theta+Pea*np.sqrt(Theta))
C1 = 1.25
C2 = 0
B0[0] = 0
B0[1:] = A*np.exp(-np.sqrt(Theta)*x[1:])+A2*np.exp(np.sqrt(Theta)*x[1:])+C1*np.exp(-Pea*x[1:])+C2

# Set up the figure and axis
fig, ax = plt.subplots()
line, = ax.plot(x, B, label='B(t)')
ax.set_xlabel('gamma=z/L')
ax.set_ylabel('B')
ax.set_title('Numerical solution to B along with analytical steady state')
ax.set_ylim(min(B0) - 0.1, max(B0) + 0.1)

# Text annotation for timestep
time_text = ax.text(
    0.5, 0.9, '',
    transform=ax.transAxes,
    fontsize=12,
    color='darkred',
    ha='center'
)
print("per-step source increment ~", dt * Theta * np.max(steady_stateA))


print("CFL",Dx**2/(2*D),"and for advection", +U*dt/Dx, "should be <0.5 for stability")

# Evaluate and plot the analytical steady state on the same axes
ana_line, = ax.plot(x, B0, 'k--', lw=2, label='analytical steady state')
ax.legend()


def update(frame):
    global B
    B_new = B.copy()
    for i in range(1, Nx-1):
        # central difference for diffusion and backward difference for advection (upwind scheme for stability in strong advection)
        B_new[i] = (B[i] + dt*(D * ((B[i+1]-2*B[i]+B[i-1])/(Dx**2)) + Pea * (B[i] - B[i-1]) /Dx
        + Theta*(steady_stateA[i])))
    # Apply boundary conditions
    B_new[0] = 0  
    B_new[-1] = (D/Dx * B[-2]) / (U + D/Dx) # Robin Boundary condition, no flux
    
    
    B = B_new
    line.set_ydata(B)

    # analytical steady-state is time-independent, keep plotted
    ana_line.set_ydata(B0)


    # Update timestep display
    time_text.set_text(f"t = {frame * dt:.1f} s")


    return line, time_text, ana_line

# Create animation
ani = FuncAnimation(fig, update, frames=range(0, Nt, 1000), interval=1, repeat=False)
plt.rcParams['animation.embed_limit'] = 2**128
plt.show()


