import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_heat_equation_explicit(alpha, dx, dy, dt, nx, ny, nt):
    """
    Solves the 2D heat equation over a rectangular domain using an explicit finite difference method.

    The function simulates heat diffusion in a two-dimensional space where the initial
    temperature distribution is specified as 100 units in a central square region, and
    the boundary conditions are implicitly set to 0 (Dirichlet boundary conditions) by
    the nature of the Laplacian computation and the array's default initialization.

    Parameters:
    - alpha (float): The thermal diffusivity of the material.
    - dx (float): The spacing between grid points in the x-direction.
    - dy (float): The spacing between grid points in the y-direction.
    - dt (float): The time step for the simulation.
    - nx (int): The number of grid points in the x-direction.
    - ny (int): The number of grid points in the y-direction.
    - nt (int): The number of time steps to simulate.

    Returns:
    - u (ndarray): A 2D numpy array of size (nx, ny), containing the simulated temperature distribution
      after nt time steps.

    Example:
    >>> alpha = 0.01
    >>> dx = dy = 0.1
    >>> dt = 0.01
    >>> nx = ny = 50
    >>> nt = 100
    >>> u = solve_heat_equation(alpha, dx, dy, dt, nx, ny, nt)
    >>> u.shape
    (50, 50)
    """
    u = np.zeros((nx, ny))
    u[int(nx / 4):int(3 * nx / 4), int(ny / 4):int(3 * ny / 4)] = 100  # Initial condition

    def laplacian(u):
        return (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) - 2 * u) / dx**2 + \
               (np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 2 * u) / dy**2

    for _ in range(nt):
        u += alpha * dt * laplacian(u)
    return u