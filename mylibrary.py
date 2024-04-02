import numpy as np
from matplotlib import pyplot as plt

# Codes for numerical solutions

"""
Get the matrices A and B for solving the diffusion equation using Crank-Nicolson method.
This function is used for vacuum boundary conditions.

Parameters:
- N: Number of spatial grid points
- sigma: alpha*dt/dx^2

Returns:
- A: Matrix A
- B: Matrix B
"""

def diff_mat_dirichlet(N, sigma):
    # Initialize matrices A and B with zeros
    A = [[0] * N for _ in range(N)]
    B = [[0] * N for _ in range(N)]

    # Interior points
    for i in range(0, N):
        A[i][i] = 2 + 2 * sigma  # Diagonal element of A
        B[i][i] = 2 - 2 * sigma  # Diagonal element of B

        # Connect to the left neighbor (if not on the left edge)
        if i > 0:
            A[i][i - 1] = -sigma
            B[i][i - 1] = sigma

        # Connect to the right neighbor (if not on the right edge)
        if i < N - 1:
            A[i][i + 1] = -sigma
            B[i][i + 1] = sigma

    return A, B

"""
Get the matrices A and B for solving the diffusion equation using Crank-Nicolson method.
This function is used for isolated boundary conditions.

Parameters:
- N: Number of spatial grid points
- sigma: alpha*dt/dx^2

Returns:
- A: Matrix A
- B: Matrix B
"""

def diff_matrix_isolated_boundary(N, sigma):
    # Initialize matrices A and B with zeros
    A = [[0] * N for _ in range(N)]
    B = [[0] * N for _ in range(N)]

    # Fill diagonal and off-diagonal values for matrices A and B
    for i in range(N):
        A[i][i] = 2 + 2 * sigma  # Diagonal element of A
        B[i][i] = 2 - 2 * sigma  # Diagonal element of B

        # Connect to the left neighbor (if not on the left edge)
        if i > 0:
            A[i][i - 1] = -sigma
            B[i][i - 1] = sigma

        # Connect to the right neighbor (if not on the right edge)
        if i < N - 1:
            A[i][i + 1] = -sigma
            B[i][i + 1] = sigma

    # Boundary conditions
    A[0][0] = 2 + sigma
    B[0][0] = 2 - sigma
    A[-1][-1] = 2 + sigma
    B[-1][-1] = 2 - sigma

    return A, B

"""
Solve 1D diffusion equation using Crank-Nicolson method.

Parameters:
- x_max: Extent of the spatial domain
- t_max: Total simulation time
- dx: Spatial step size
- dt: Time step size
- Diff: Thermal diffusivity
- init_cond: Initial condition function
- source_term: Source term function
- boundary: Boundary condition function

Returns:
- u: Temperature distribution over space and time
- x: Spatial grid
- t: Time grid
"""

def CN_diffusion(x_min, x_max, t_max, dx, dt, Diff, init_cond, source_term, boundary):

    alpha = Diff * dt / (dx**2)

    # Spatial grid
    x = np.linspace(x_min, x_max, int((x_max - x_min) / dx) + 1)
    t = np.linspace(0, t_max, int(t_max / dt) + 1)

    # Initialize temperature array
    Temp = np.zeros((len(x), len(t)))

    # Initial condition
    for i in range(len(x)):
        Temp[i][0] = init_cond(x[i])

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = boundary(len(x), alpha)

    A = np.array(A)
    B = np.array(B)

    for j in range(1, len(t)):
        source_vector = np.array([source_term(xi, t[j]) for xi in x])
        Temp[:, j] = np.linalg.solve(A, np.dot(B, Temp[:, j - 1]) + dt * source_vector)

    return Temp, np.array(x), np.array(t)

"""
Calculation of the pitch angle

Parameters:
- Br: Radial component of the magnetic field
- Bphi: Azimuthal component of the magnetic field

Returns:
- B: Total magnetic field
- p: Pitch angle
"""

def get_B_and_pitch(Br, Bphi):
    B = np.sqrt(Br**2 + Bphi**2)
    p = np.zeros(Br.shape)
    for i in range(Br.shape[0]):
        for j in range(Br.shape[1]):
            if Bphi[i, j]!=0:
                p[i, j] = 180/np.pi*np.arctan(Br[i, j]/Bphi[i, j])
            elif Br[i, j]>0:
                p[i, j] = 90
            elif Br[i, j]<0:
                p[i, j] = -90
            else:
                p[i, j] = 0
    return B, p

"""
Get the matrices A and B for solving the diffusion equation using Crank-Nicolson method.
This function is used for vacuum boundary conditions.

Parameters:
- N: Number of spatial grid points
- a1, b1, ... etc: Coefficients of the matrix

Returns:
- A: Matrix A
- B: Matrix B
"""

def matrix_A(N, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4):
    # return a 2N x 2N matrix with the given terms in each block of the matrix
    A = np.zeros((2*N, 2*N))
    for i in range(N):
        A[i, i] = a1
        A[i, i+N] = a2
        A[i+N, i] = a3
        A[i+N, i+N] = a4
    for i in range(N-1):
        A[i, i+1] = b1
        A[i, i+N+1] = b2
        A[i+N, i+1] = b3
        A[i+N, i+N+1] = b4
        A[i+1, i] = c1
        A[i+1, i+N] = c2
        A[i+N+1, i] = c3
        A[i+N+1, i+N] = c4
    return A

def matrix_B(N, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4):
    B = np.zeros((2*N, 2*N))
    for i in range(N):
        B[i, i] = a1
        B[i, i+N] = a2
        B[i+N, i] = a3
        B[i+N, i+N] = a4
    for i in range(N-1):
        B[i, i+1] = b1
        B[i, i+N+1] = b2
        B[i+N, i+1] = b3
        B[i+N, i+N+1] = b4
        B[i+1, i] = c1
        B[i+1, i+N] = c2
        B[i+N+1, i] = c3
        B[i+N+1, i+N] = c4
    return B

"""
Solve 1D diffusion equation using Crank-Nicolson method with modified matrices.

Parameters:
- N_x: Number of spatial grid points
- N_t: Number of time grid points
- init_cond_Br: Initial condition for Br
- init_cond_Bphi: Initial condition for Bphi
- A: Coefficient matrix A
- B: Coefficient matrix B

Returns:
- U: Magnetic field distribution over space and time
"""

def crank_nicolson_mod(N_x, N_t, init_cond_Br, init_cond_Bphi, A, B):

    # Initialize temperature array
    U = np.zeros((2*N_x, N_t))

    # Initial condition
    for i in range(N_x):
        U[i, 0] = init_cond_Br[i]
        U[N_x+i, 0] = init_cond_Bphi[i]

    for j in range(1, N_t):
        U[:, j] = np.dot(np.linalg.inv(A), np.dot(B, U[:, j - 1]))

    return U






# Codes for plotting functions

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def create_animation(B_array, z_values, t_values, filename='animation.gif', B_label='B(z)', z_label='z'):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.plot(z_values, B_array[:, frame])
        ax.set_title(f"Time = {t_values[frame]:.2f}")
        ax.set_xlabel(z_label)
        ax.set_ylabel(B_label)
        ax.set_xlim(z_values.min(), z_values.max())
        ax.set_ylim(B_array.min(), B_array.max())

    ani = animation.FuncAnimation(fig, update, frames=len(t_values), interval=100)

    ani.save(filename, writer='pillow')
    plt.close(fig)


def plot_init_cond(z, init_cond_Br, init_cond_Bphi, title1, title2, global_title):
    plt.figure(figsize=(11, 3.5))
    plt.subplot(121)
    plt.plot(z, init_cond_Br(z))
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel(r'$B_r$')
    plt.title(title1)

    plt.subplot(122)
    plt.plot(z, init_cond_Bphi(z))
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel(r'$B_{\phi}$')
    plt.title(title2)

    plt.suptitle(global_title)
    plt.tight_layout(pad=1)


"""
Plot the solution in both 1D and Heatmap format.

Parameters:
- time_grid: Time grid
- spatial_grid: Spatial grid
- solution: Solution of the diffusion equation

Returns:
- Makes the plots
"""

def plot_diff(time_grid, spatial_grid, solution_r, solution_phi):

    # Create 2D plots
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    for i in (range(0, len(time_grid), int(len(time_grid)/5))):
        plt.plot(spatial_grid, solution_r[:, i], label=f'time = {time_grid[i]:.1f}')
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel(r'Magnetic Field Strength ($B_r$)')
    plt.title('Diffusion of Magnetic field in radial direction')
    # plt.ylim(np.min(solution_r), np.max(solution_r))
    plt.grid()
    plt.legend()

    # Create imshow plot
    plt.subplot(2, 2, 2)
    plt.contourf(*np.meshgrid(spatial_grid, time_grid), solution_r.T, 50, cmap='Spectral_r')
    plt.colorbar(label=r'Magnetic Field Strength ($B_r$)')
    plt.title(r'Diffusion of Magnetic field in radial direction')
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel(r'Time (Myr)')


    # Create 2D plots
    plt.subplot(2, 2, 3)
    for i in (range(0, len(time_grid), int(len(time_grid)/5))):
        plt.plot(spatial_grid, solution_phi[:, i], label=f'time = {time_grid[i]:.1f}')
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel(r'Magnetic Field Strength ($B_\phi$)')
    plt.title(r'Diffusion of Magnetic field in azimuthal direction')
    # plt.ylim(np.min(solution_phi), np.max(solution_phi))
    plt.grid()
    plt.legend()

    # Create imshow plot
    plt.subplot(2, 2, 4)
    plt.contourf(*np.meshgrid(spatial_grid, time_grid), solution_phi.T, 50, cmap='Spectral_r')
    plt.colorbar(label=r'Magnetic Field Strength ($B_\phi$)')
    plt.title('Diffusion of Magnetic field in azimuthal direction')
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel('Time (Myr)')
    plt.tight_layout(pad=3)



def plot_pitch(time_grid, spatial_grid, B, pitch):

    # Create 2D plots
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    for i in (range(0, len(time_grid), int(len(time_grid)/5))):
        plt.plot(spatial_grid, B[:, i], label=f'time = {time_grid[i]:.1f}')
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel(r'$B_{total}$')
    plt.title('Diffusion of total magnetic field')
    plt.grid()
    plt.legend()

    # Create imshow plot
    plt.subplot(2, 2, 2)
    plt.contourf(*np.meshgrid(spatial_grid, time_grid), B.T, 40, cmap='Spectral_r')
    plt.colorbar(label=r'($B_{total}$)')
    plt.title(r'Diffusion of total magnetic field')
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel(r'Time (Myr)')


    # Create 2D plots
    plt.subplot(2, 2, 3)
    for i in (range(0, len(time_grid), int(len(time_grid)/5))):
        plt.plot(spatial_grid[1:-1], pitch[1:-1, i], label=f'time = {time_grid[i]:.1f}')
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel(r'Pitch angle $p_B$ (in degrees)')
    plt.title(r'Variation of pitch angle with time')
    plt.grid()
    plt.legend()

    # Create imshow plot
    plt.subplot(2, 2, 4)
    plt.contourf(*np.meshgrid(spatial_grid[1:-1], time_grid), pitch.T[:, 1:-1], 40, cmap='Spectral_r')
    plt.colorbar(label=r'Pitch angle $p_B$ (in degrees)')
    plt.title('Variation of pitch angle with time')
    plt.xlabel(r'$z$ (normalized to 100 pc)')
    plt.ylabel('Time (Myr)')

    plt.tight_layout(pad=3)



def plot_decay(time_grid, B_mid, m, c):
    # Plot the log of magnetic field strength at midplane and the slope of the logplot
    plt.figure(figsize=(6, 4))
    plt.plot(time_grid, B_mid, 'b-')
    # plot another line with the slope and intercept m and c
    plt.plot(time_grid[-50:], m*time_grid[-50:] + c, 'r:', linewidth=3, label=r'Slope ($\gamma$) = {:.3e}'.format(m))
    plt.xlabel('Time (Myr)')
    plt.ylabel('log$(B_{total})$ at midplane')
    plt.title(r'Magnetic field strength at midplane')
    # plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.tight_layout()