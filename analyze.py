"""Tools to analyze the latent space."""

import csv
import numpy as np
import torch

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

import train_utils

DEVICE = 'cuda'

N_PCA_COMPONENTS = 5


def latent_projection(output, dataset_path, algo_name='vae', epoch_id=None):
    ckpt = train_utils.load_checkpoint(
        output=output, algo_name=algo_name, epoch_id=epoch_id)
    spd_feature = ckpt['nn_architecture']['spd_feature']

    dataset = np.load(dataset_path)
    if spd_feature is not None:
        dataset = train_utils.spd_feature_from_matrix(
            dataset, spd_feature=spd_feature)

    encoder = train_utils.load_module(
        output, module_name='encoder', epoch_id=epoch_id)
    dataset = torch.Tensor(dataset)
    dataset = torch.utils.data.TensorDataset(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    mus = []
    for i, data in enumerate(loader):
        data = data[0].to(DEVICE)  # extract from loader's list
        if len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=0)
        assert len(data.shape) == 4
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


def get_subset_fmri(output, metadata_csv, ses_ids=None, task_names=None,
                    epoch_id=None, n_pcs=2):
    paths_subset = []

    tasks_subset = []
    ses_subset = []
    times_subset = []

    with open(metadata_csv, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            path, ses, task, run, time = row
            if path == 'path':
                continue
            ses = int(ses)
            run = int(run)
            time = int(time)
            if (task_names is not None) and (task not in task_names):
                continue
            if run != 1:
                # Only take one run per session
                continue
            if (ses_ids is not None) and (ses not in ses_ids):
                continue

            paths_subset.append(path)

            tasks_subset.append(task)
            ses_subset.append(ses)
            times_subset.append(time)

    subset = [np.load(one_path) for one_path in paths_subset]
    subset = np.array(subset)

    # Note: dataset needs to be unshuffled here
    mus = latent_projection(output, subset, epoch_id=epoch_id)
    _, projected_mus = pca_projection(mus, n_pcs)

    labels_subset = {
        'task': tasks_subset, 'ses': ses_subset, 'time': times_subset}

    return projected_mus, labels_subset
