import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SpaCoObject import SPACO


def plot_spatial_heatmap(
    coords, values, title="Spatial Heatmap", cmap="viridis", point_size=50
):
    """
    Plots a discrete heatmap based on coordinates and corresponding values.
    Args:
        coords (numpy.ndarray): Array of shape (n_samples, 2) containing x and y coordinates.
        values (numpy.ndarray): Array of shape (n_samples,) containing values corresponding to the coordinates.
        title (str): Title of the plot.
        cmap (str): Colormap to use for the heatmap.
        point_size (int): Size of the points in the scatter plot.
    """
    # Ensure inputs are NumPy arrays
    coords = np.array(coords)
    values = np.array(values)
    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=point_size, edgecolor="k"
    )
    plt.colorbar(scatter, label="Values")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


def data_loader():
    # testing the spaco_py package
    # Read in the data
    sample_features = pd.read_excel("./mouseBrainTestData/SF_brain.xlsx").to_numpy()
    neighborhood = pd.read_excel("./mouseBrainTestData/A_brain.xlsx").to_numpy()

    # need coordinates for the plots
    coordinates = pd.read_excel("./mouseBrainTestData/coords_brain.xlsx").to_numpy()

    spaco = SPACO(sample_features, neighborhood)
    return spaco, sample_features, neighborhood, coordinates


if __name__ == "__main__":
    spaco_object, sample_features, neighborhood, coordinates = data_loader()

    breakpoint = 0
