import numpy as np 

def solve_poisson_equation_gs(f, dx, dy, nx, ny):
    """
    Solves the 2D Poisson equation over a rectangular domain using the Gauss-Seidel iterative method.

    The function finds the steady-state solution to the Poisson equation given a source term,
    assuming Dirichlet boundary conditions of u=0 on all boundaries. The Gauss-Seidel method is an
    iterative algorithm for solving systems of linear equations that updates the solution using the
    most recently calculated values, which can lead to faster convergence compared to the Jacobi method.

    Parameters:
    - f (ndarray): A 2D numpy array of size (nx, ny) representing the source term of the Poisson equation.
    - dx (float): The spacing between grid points in the x-direction.
    - dy (float): The spacing between grid points in the y-direction.
    - nx (int): The number of grid points in the x-direction.
    - ny (int): The number of grid points in the y-direction.

    Returns:
    - u (ndarray): A 2D numpy array of size (nx, ny), containing the potential field that satisfies
      the Poisson equation for the given source term f and boundary conditions.
    """
    u = np.zeros((nx, ny))

    for _ in range(10000):  # Iterative solver loop
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                u[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] +
                                  u[i, j-1] + u[i, j+1] -
                                  dx**2 * f[i, j])
    return u
