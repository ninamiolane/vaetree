"""Tools to analyze the latent space."""

import numpy as np
import torch

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

import vis

DEVICE = 'cuda'

N_PCA_COMPONENTS = 5


def latent_projection(output_dir, dataset):
    encoder = vis.load_last_module(output_dir, 'encoder')
    dataset = torch.Tensor(dataset)
    dataset = torch.utils.data.TensorDataset(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    mus = []
    for i, data in enumerate(loader):
        data = data[0].to(DEVICE)
        mu, logvar = encoder(data)
        mus.append(np.array(mu.cpu().detach()))

    mus = np.array(mus).squeeze()
    return mus


def pca_projection(mus, n_pca_components=N_PCA_COMPONENTS):
    pca = PCA(n_components=n_pca_components)
    pca.fit(mus)
    projected_mus = pca.transform(mus)
    return pca, projected_mus


def plot_kde(ax, projected_mus):
    x = projected_mus[:, 0]
    y = projected_mus[:, 1]
    data = np.vstack([x, y])
    kde = gaussian_kde(data)

    # Evaluate on a regular grid
    xgrid = np.linspace(-4, 4, 200)
    ygrid = np.linspace(-5, 5, 200)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    # Plot the result as an image
    ax.imshow(Z.reshape(Xgrid.shape),
              origin='lower', aspect='auto',
              extent=[-4, 4, -5, 5],
              cmap='Blues')
    return ax
