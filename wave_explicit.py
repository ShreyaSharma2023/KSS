import numpy as np

def solve_wave_equation_explicit(c, dt, dx, dy, nx, ny, nt):
    """
    Solves the 2D wave equation over a rectangular domain using a finite difference method.

    This function simulates the propagation of waves in a two-dimensional space
    under the assumption of an initial displacement in the central region of the domain.
    The simulation uses fixed boundary conditions (u=0) at the domain edges by not updating
    the boundary grid points. The method employed is an explicit scheme, specifically
    a second-order central difference in both space and time.

    Parameters:
    - c (float): The wave speed.
    - dt (float): The time step for the simulation.
    - dx (float): The spacing between grid points in the x-direction.
    - dy (float): The spacing between grid points in the y-direction.
    - nx (int): The number of grid points in the x-direction.
    - ny (int): The number of grid points in the y-direction.
    - nt (int): The number of time steps to simulate.

    Returns:
    - u (ndarray): A 2D numpy array of size (nx, ny), containing the simulated wave displacement
      after nt time steps.

    Example:
    >>> c = 1.0
    >>> dx = dy = 0.1
    >>> dt = 0.01
    >>> nx = ny = 50
    >>> nt = 100
    >>> u = solve_wave_equation(c, dt, dx, dy, nx, ny, nt)
    >>> u.shape
    (50, 50)
    """
    u = np.zeros((nx, ny))
    u_new = np.zeros((nx, ny))
    u_old = np.zeros((nx, ny))
    u[int(nx / 4):int(3 * nx / 4), int(ny / 4):int(3 * ny / 4)] = 1  # Initial displacement

    for _ in range(nt):
        u_new[1:-1, 1:-1] = (2 * u[1:-1, 1:-1] - u_old[1:-1, 1:-1] +
                             c**2 * dt**2 * ((u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
                                            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2))
        u_old, u = u, u_new.copy()
    return u
