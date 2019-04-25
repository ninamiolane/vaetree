""" Visualization for toy experiments."""

import numpy as np
import seaborn as sns

from scipy.stats import gaussian_kde

ALPHA = 0.2
BINS = 40


def plot_data(x_data, color='darkgreen', label=None, ax=None):
    _, data_dim = x_data.shape
    if data_dim == 1:
        ax.hist(x_data, bins=BINS, alpha=ALPHA, color=color, label=label, density=True)
    else:
        sns.scatterplot(x_data[:, 0], x_data[:, 1], ax=ax, color=color)
    return ax


def plot_kde(some_x, ax, cmap, axis_side):
    x = some_x[:, 0]
    y = some_x[:, 1]
    data = np.vstack([x, y])
    kde = gaussian_kde(data)

    # evaluate on a regular grid
    xgrid = np.linspace(-axis_side, axis_side, 200)
    ygrid = np.linspace(-axis_side, axis_side, 200)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    # Plot the result as an image
    ax.imshow(
        Z.reshape(Xgrid.shape),
        origin='lower', aspect='auto',
        extent=[-2, 2, -2, 2],
        cmap=cmap)
