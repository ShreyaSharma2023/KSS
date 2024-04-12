import numpy as np
import matplotlib.pyplot as plt


# Parameters
nx, ny = 50, 50  # Domain extent
dx = dy = 0.1    # Mesh spacing
dt = 0.01        # Time step 
nt = 100         # Number of time steps
alpha = 0.01     # Thermal diffusivity constant
c = 1.0          # Wave speed

# Computational Domain
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Forcing function for Poisson's Eq.
f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)



