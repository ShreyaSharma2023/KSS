import matplotlib.pyplot as plt

def plot_solution(solution, title, colormap, extent):
    """
    Plots the solution matrix of a PDE.

    Parameters:
    - solution: 2D numpy array representing the solution to plot.
    - title: String representing the title for the plot.
    - colormap: Colormap to use for the plot.
    - extent: The bounding box in data coordinates that the image will fit into.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(solution, extent=extent, origin='lower', cmap=colormap)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()