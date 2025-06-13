import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_array(arr):
    """
    Plots a 3D scatter of all points in an 8×8×4 array,
    coloring each point by its value.
    """
    # Generate integer coordinates for each element
    x, y, z = np.indices(arr.shape)

    # Flatten the coordinates and values for scatter
    xs = x.flatten()
    ys = y.flatten()
    zs = z.flatten()
    values = arr.flatten()

    # Create the figure and a 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    sc = ax.scatter(xs, ys, zs, c=values, marker='o')

    # Add a colorbar
    cb = fig.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label('Value')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Optionally adjust the viewing angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example: replace this with your actual 8×8×4 data array
    sample_array = np.random.rand(8, 8, 4)
    plot_3d_array(sample_array)
