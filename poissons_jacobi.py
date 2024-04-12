import numpy as np

def solve_poisson_equation_jacobi(f, dx, dy, nx, ny):
    """
    Solves the 2D Poisson equation over a rectangular domain using the Jacobi iterative method.

    The function finds the steady-state solution to the Poisson equation given a source term,
    assuming Dirichlet boundary conditions of u=0 on all boundaries. The Jacobi method is an
    iterative algorithm for solving systems of linear equations, which is particularly suited
    for discretized differential equations like the Poisson equation.

    Parameters:
    - f (ndarray): A 2D numpy array of size (nx, ny) representing the source term of the Poisson equation.
    - dx (float): The spacing between grid points in the x-direction.
    - dy (float): The spacing between grid points in the y-direction.
    - nx (int): The number of grid points in the x-direction.
    - ny (int): The number of grid points in the y-direction.

    Returns:
    - u (ndarray): A 2D numpy array of size (nx, ny), containing the potential field that satisfies
      the Poisson equation for the given source term f and boundary conditions.

    Example:
    >>> nx, ny = 50, 50
    >>> dx = dy = 0.1
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> X, Y = np.meshgrid(x, y)
    >>> f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    >>> u = solve_poisson_equation(f, dx, dy, nx, ny)
    >>> u.shape
    (50, 50)
    """
    u = np.zeros((nx, ny))

    for _ in range(10000):  # Iterative solver loop
        u_old = u.copy()
        u[1:-1, 1:-1] = 0.25 * (u_old[:-2, 1:-1] + u_old[2:, 1:-1] +
                                u_old[1:-1, :-2] + u_old[1:-1, 2:] -
                                dx**2 * f[1:-1, 1:-1])
    return u