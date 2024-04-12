import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_heat_equation_implicit(alpha, dx, dy, dt, nx, ny, nt):
    """
    Solves the 2D heat equation using an implicit finite difference method with Dirichlet boundary conditions.
    """
    # Initialize the temperature distribution
    u = np.zeros((nx, ny))
    u[int(nx / 4):int(3 * nx / 4), int(ny / 4):int(3 * ny / 4)] = 100

    # Number of inner points
    nx_inner, ny_inner = nx - 2, ny - 2

    # Coefficients for the matrix
    lambda_x = alpha * dt / dx**2
    lambda_y = alpha * dt / dy**2

    # Generate the sparse matrix for the implicit scheme
    main_diag = (1 + 2 * lambda_x + 2 * lambda_y) * np.ones(nx_inner * ny_inner)
    off_diag_x = -lambda_x * np.ones(nx_inner * ny_inner - 1)
    off_diag_y = -lambda_y * np.ones(nx_inner * ny_inner - nx_inner)
    
    # Avoid connecting the end of a row with the beginning of the next one
    for i in range(1, ny_inner):
        off_diag_x[i * nx_inner - 1] = 0
    
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    A = diags(diagonals, [0, -1, 1, -nx_inner, nx_inner], format="csr")

    # Flatten the inner part of the temperature matrix for linear system solving
    u_inner = u[1:-1, 1:-1].flatten()

    # Time-stepping
    for _ in range(nt):
        # Solve the linear system
        u_inner = spsolve(A, u_inner)

        # Update the temperature distribution (while keeping boundary conditions)
        u[1:-1, 1:-1] = u_inner.reshape((nx_inner, ny_inner))

    return u