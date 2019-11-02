"""Tools to analyze the latent space."""

import csv
import functools
import importlib
import os

import numpy as np
import ot
import torch

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from geomstats.geometry.discretized_curves_space import DiscretizedCurvesSpace


import geom_utils
import train_utils

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

N_PCA_COMPONENTS = 5

TOY_LOGVARX_TRUE = [-10, -5, -3.22, -2, -1.02, -0.45, 0]
TOY_STD_TRUE = np.sqrt(np.exp(TOY_LOGVARX_TRUE))
TOY_N = [10000, 100000]


def latent_projection(output, dataset_path, algo_name='vae', epoch_id=None):
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)
    dataset = np.load(dataset_path)

    if 'spd_feature' in ckpt['nn_architecture']:
        spd_feature = ckpt['nn_architecture']['spd_feature']
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


def submanifold_from_t_and_decoder_in_euclidean(
        t, decoder, logvarx=1, with_noise=False):
    """
    Generate data using generative model from decoder.
    Euclidean Gaussian noise uses the logvarx given as input.

    Logvarx is fixed, as opposed to using logvarx generated
    from z by decoder.
    """
    t = torch.Tensor(t).to(DEVICE)
    mux, _ = decoder(t)
    mux = mux.cpu().detach().numpy()
    n_samples, data_dim = mux.shape

    generated_x = mux
    if with_noise:
        generated_x = np.zeros((n_samples, data_dim))
        for i in range(n_samples):
            logvar = logvarx
            sigma = np.sqrt(np.exp((logvar)))
            eps = np.random.normal(
                loc=0, scale=sigma, size=(1, data_dim))
            generated_x[i] = mux[i] + eps

    return mux, generated_x


def submanifold_from_t_and_decoder_on_manifold(
        t, decoder, logvarx=1, manifold_name='h2', with_noise=False):
    """
    The decoder generate on the tangent space of a manifold.
    We use Exp to bring these points on the manifold.
    We add a Gaussian noise at each point.
    To this aim, we use a wrapped Gaussian: we generate a Gaussian noise
    at the tangent space of the point, and use the Exp at the point to
    get a point on the manifold.

    Logvarx is fixed, as opposed to using logvarx generated
    from z by decoder.

    Extrinsic representation for points on manifold (3D).
    """
    t = torch.Tensor(t).to(DEVICE)
    mux, _ = decoder(t)
    mux = mux.cpu().detach().numpy()
    n_samples, data_dim = mux.shape

    mux = geom_utils.convert_to_tangent_space(mux, manifold_name=manifold_name)
    manifold, base_point = geom_utils.manifold_and_base_point(manifold_name)

    mux_riem = manifold.metric.exp(mux, base_point=base_point)

    generated_x = mux_riem
    if with_noise:
        scale = np.sqrt(np.exp(logvarx))
        eps = np.random.normal(
            loc=0, scale=scale, size=(n_samples, data_dim+1))  # HACK!
        eps = manifold.projection_to_tangent_space(
            vector=eps, base_point=mux_riem)

        generated_x = manifold.metric.exp(eps, base_point=mux_riem)

    return mux_riem, generated_x


def submanifold_from_t_and_decoder_on_tangent_space(
        t, decoder, logvarx=1, manifold_name='h2', with_noise=False):
    """
    Bring the generated points back on the tangent space
    at the chosen basepoint and uses 2D..

    Logvarx is fixed, as opposed to using logvarx generated
    from z by decoder.

    Intrinsic representation for vectors on tangent space (2D).
    """
    t = torch.Tensor(t).to(DEVICE)
    mux_riem, generated_x = submanifold_from_t_and_decoder_on_manifold(
        t, decoder, logvarx, manifold_name)

    manifold, base_point = geom_utils.manifold_and_base_point(manifold_name)

    mux_riem_on_tangent_space = manifold.metric.log(
        mux_riem, base_point=base_point)
    if manifold_name == 's2':
        mux_riem_on_tangent_space = mux_riem_on_tangent_space[:, :2]
    elif manifold_name == 'h2':
        mux_riem_on_tangent_space = mux_riem_on_tangent_space[:, 1:]

    generated_x_on_tangent_space = mux_riem_on_tangent_space
    if with_noise:
        generated_x_on_tangent_space = manifold.metric.log(
            generated_x, base_point=base_point)
        if manifold_name == 's2':
            generated_x_on_tangent_space = generated_x_on_tangent_space[:, :2]
        elif manifold_name == 'h2':
            generated_x_on_tangent_space = generated_x_on_tangent_space[:, 1:]
    return mux_riem_on_tangent_space, generated_x_on_tangent_space


def true_submanifold_from_t_and_decoder(
        t, decoder, manifold_name='r2',
        with_noise=False, logvarx_true=None):
    """
    Logvarx is fixed, as opposed to using logvarx generated
    from z by decoder.
    """
    if manifold_name == 'r2':
        x_novarx, x = submanifold_from_t_and_decoder_in_euclidean(
            t, decoder, logvarx=logvarx_true, with_noise=with_noise)

    elif manifold_name in ['s2', 'h2']:
        x_novarx, x = submanifold_from_t_and_decoder_on_manifold(
            t, decoder, logvarx=logvarx_true,
            manifold_name=manifold_name, with_noise=with_noise)
    else:
        raise ValueError('Manifold not supported.')

    return x_novarx, x


def learned_submanifold_from_t_and_decoder(
        t, decoder, vae_type='gvae_tgt',
        manifold_name='r2', with_noise=False):
    """
    Logvarx is fixed to -5, as opposed to using logvarx generated
    from z by decoder.

    This is because decoders are trained with logvarx =-5.
    """
    logvarx = -5
    if manifold_name in ['r2', 'r3']:
        x_novarx, x = submanifold_from_t_and_decoder_in_euclidean(
            t, decoder, logvarx=logvarx, with_noise=with_noise)

    elif manifold_name in ['s2', 'h2']:
        x_novarx, x = submanifold_from_t_and_decoder_on_manifold(
            t, decoder, logvarx=logvarx,
            manifold_name=manifold_name, with_noise=with_noise)
    else:
        raise ValueError('Manifold not supported.')

    return x_novarx, x


def true_submanifold_from_t_and_output(
        t, output, algo_name='vae', manifold_name='r2',
        epoch_id=None, with_noise=False):
    """
    Generate:
    - true_x_no_var: true submanifold used in the experiment output
    - true_x: data generated from true model
    """
    decoder_true_path = '%s/decoder_true.pth' % output
    decoder_true = torch.load(decoder_true_path, map_location=DEVICE)

    logvarx_true = None
    if with_noise:
        ckpt = train_utils.load_checkpoint(
            output, epoch_id=20)
        logvarx_true = ckpt['nn_architecture']['logvarx_true']

    true_x_novarx, true_x = true_submanifold_from_t_and_decoder(
        t, decoder_true, manifold_name=manifold_name,
        with_noise=with_noise, logvarx_true=logvarx_true)
    return true_x_novarx, true_x


def learned_submanifold_from_t_and_output(
        t, output, algo_name='vae', manifold_name='r2',
        epoch_id=None, with_noise=False):
    """
    Generate:
    - true_x_no_var: true submanifold used in the experiment output
    - true_x: data generated from true model
    """
    decoder = train_utils.load_module(
        output, module_name='decoder', epoch_id=epoch_id)

    # logvarx_true = None
    # if with_noise:
    #    ckpt = train_utils.load_checkpoint(
    #        output, algo_name=algo_name, epoch_id=epoch_id)
    #    logvarx_true = ckpt['nn_architecture']['logvarx_true']
    #    # TODO(nina): Decide if using the truth or cst -5

    x_novarx, x = learned_submanifold_from_t_and_decoder(
        t, decoder, manifold_name=manifold_name,
        with_noise=with_noise)
    return x_novarx, x


def learned_submanifold_from_t_and_vae_type(
        t, vae_type, logvarx_true, n,
        algo_name='vae', manifold_name='r2',
        epoch_id=None, with_noise=False):
    """
    Generate:
    - true_x_no_var: true submanifold used in the experiment output
    - true_x: data generated from true model
    """
    train_dict = {}
    train_dict['algo_name'] = algo_name
    train_dict['manifold_name'] = manifold_name
    train_dict['vae_type'] = vae_type
    train_dict['logvarx_true'] = logvarx_true
    train_dict['n'] = n

    if vae_type in ['gvae', 'gvae_tgt']:
        output = ray_output_dir(train_dict)

        x_novarx, x = learned_submanifold_from_t_and_output(
            t, output=output,
            algo_name=algo_name, manifold_name=manifold_name,
            epoch_id=epoch_id, with_noise=with_noise)
    elif vae_type == 'vae':
        output = ray_output_dir(train_dict)

        x_novarx, x = learned_submanifold_from_t_and_output(
            t, output=output,
            algo_name=algo_name, manifold_name='r3',
            epoch_id=epoch_id, with_noise=with_noise)
    elif vae_type == 'vae_proj':
        train_dict['vae_type'] = 'vae'
        output = ray_output_dir(train_dict)

        x_novarx, x = learned_submanifold_from_t_and_output(
            t, output=output,
            algo_name=algo_name, manifold_name='r3',
            epoch_id=epoch_id, with_noise=with_noise)

        norms = np.linalg.norm(x_novarx, axis=1)
        norms = np.expand_dims(norms, axis=1)
        x_novarx = x_novarx / norms

        norms = np.linalg.norm(x, axis=1)
        norms = np.expand_dims(norms, axis=1)
        x = x / norms
    elif vae_type == 'pga':
        train_dict['vae_type'] = 'gvae_tgt'
        output = ray_output_dir(train_dict)

        synthetic_dataset_in_tgt = np.load(os.path.join(
            output, 'synthetic/dataset.npy'))
        pca = PCA(n_components=1)
        pca.fit(synthetic_dataset_in_tgt)

        component_extrinsic = geom_utils.convert_to_tangent_space(
                pca.components_, manifold_name='s2')
        manifold, base_point = geom_utils.manifold_and_base_point(
                manifold_name)
        geodesic = manifold.metric.geodesic(
            initial_point=base_point, initial_tangent_vec=component_extrinsic)

        x_novarx = geodesic(t)
        x = x_novarx

    return x_novarx, x


def toyoutput_dir(manifold_name, logvarx_true, n, vae_type='gvae'):
    main_dir = '/scratch/users/nmiolane/toyoutput_manifold_%s' % vae_type
    output = os.path.join(main_dir, 'logvarx_%s_n_%d_%s' % (
            logvarx_true, n, manifold_name))
    return output


def parse_train_dir(train_dir):
    train_dir = train_dir.replace('=', '_').replace(',', '_')
    train_dict = {}
    for param in ['_algo_name_', '_manifold_name_', '_vae_type_']:
        substring = train_dir[train_dir.find(param)+len(param):]
        param_value = substring.split('_')[0]
        train_dict[param[1:-1]] = param_value
    for param in ['_logvarx_true_', '_n_']:
        substring = train_dir[train_dir.find(param)+len(param):]
        param_value = substring.split('_')[0]
    return train_dict


def ray_output_dir(train_dict, main_dir='../../ray_results/Train'):
    output_dirs = []
    for _, train_dirs, _ in os.walk(main_dir):
        for train_dir in train_dirs:
            keep_dir = True
            for param_name, param_value in train_dict.items():
                pattern = param_name + '=' + str(param_value)
                if pattern not in train_dir:
                    keep_dir = False
                    break
            if not keep_dir:
                continue
            output_dirs.append(os.path.join(main_dir, train_dir))
    if len(output_dirs) > 1:
        print('Found more than 1 dir, returning last one.')
    return output_dirs[-1]


def squared_dist_between_submanifolds(manifold_name,
                                      vae_type='gvae_tgt',
                                      all_logvarx_true=TOY_LOGVARX_TRUE,
                                      all_n=TOY_N,
                                      epoch_id=100,
                                      n_samples=1000,
                                      extrinsic_or_intrinsic='extrinsic'):
    """
    Compute:
    d(N1, N2) = int d^2(f_theta1(z), f_theta2(z)) dmu(z)
    by Monte-Carlo approximation,
    when:
    - mu is the standard normal on the 1D latent space,
    - d is the extrinsic (r3) or intrinsic dist on manifold_name
    """
    dists = np.zeros((len(all_n), len(all_logvarx_true)))

    t = np.random.normal(size=(n_samples,))

    for i_logvarx_true, logvarx_true in enumerate(all_logvarx_true):
        for i_n, n in enumerate(all_n):
            output_decoder_true = toyoutput_dir(
                vae_type='gvae_tgt', manifold_name=manifold_name,
                logvarx_true=logvarx_true, n=n)
            submanifold_true, _ = true_submanifold_from_t_and_output(
                t, output=output_decoder_true, manifold_name=manifold_name)

            submanifold_learned, _ = learned_submanifold_from_t_and_vae_type(
                t=t, vae_type=vae_type,
                logvarx_true=logvarx_true, n=n,
                manifold_name=manifold_name, epoch_id=epoch_id)

            if extrinsic_or_intrinsic == 'intrinsic':
                curves_space = DiscretizedCurvesSpace(
                    ambient_manifold=geom_utils.MANIFOLD[manifold_name])
                curves_space_metric = curves_space.l2_metric

                dist = curves_space_metric.dist(
                    submanifold_true, submanifold_learned) ** 2
            else:
                dist = np.linalg.norm(
                    submanifold_learned - submanifold_true) ** 2

            dists[i_n, i_logvarx_true] = dist / n_samples
    return dists


def make_gauss_hist(n_bins, m=0, s=1):
    x_bin_min = m - 3 * s
    x_bin_max = m + 3 * s
    x = np.linspace(x_bin_min, x_bin_max, n_bins, dtype=np.float64)
    h = np.exp(-(x - m)**2 / (2 * s**2))
    return x, h / h.sum()


def squared_w2_between_submanifolds(manifold_name,
                                    vae_type='gvae_tgt',
                                    epoch_id=100,
                                    all_logvarx_true=TOY_LOGVARX_TRUE,
                                    all_n=TOY_N,
                                    extrinsic_or_intrinsic='extrinsic',
                                    n_bins=5,
                                    sinkhorn=False):
    manifold, base_point = geom_utils.manifold_and_base_point(
        manifold_name)

    w2_dists = np.zeros((len(all_n), len(all_logvarx_true)))

    x_bins_a, a = make_gauss_hist(n_bins, m=0, s=1)
    x_bins_b, b = make_gauss_hist(n_bins, m=0, s=1)
    assert np.all(x_bins_a == x_bins_b)
    x = x_bins_a
    train_dict = {}
    train_dict['vae_type'] = vae_type
    train_dict['manifold_name'] = manifold_name

    for i_logvarx_true, logvarx_true in enumerate(all_logvarx_true):
        for i_n, n in enumerate(all_n):
            train_dict['n'] = n
            train_dict['logvarx_true'] = logvarx_true
            train_dict['vae_type'] = 'gvae_tgt'

            output_decoder_true = ray_output_dir(train_dict)

            M2 = np.zeros((n_bins, n_bins))
            for i in range(n_bins):
                for j in range(n_bins):
                    zi = np.expand_dims(np.expand_dims(x[i], axis=0), axis=1)
                    xi, _ = true_submanifold_from_t_and_output(
                        t=zi, output=output_decoder_true,
                        manifold_name=manifold_name)

                    zj = np.expand_dims(np.expand_dims(x[j], axis=0), axis=1)
                    xj, _ = learned_submanifold_from_t_and_vae_type(
                        t=zj, manifold_name=manifold_name, vae_type=vae_type,
                        logvarx_true=logvarx_true, n=n)

                    if extrinsic_or_intrinsic == 'intrinsic':
                        sq_dist = manifold.metric.squared_dist(xi, xj)
                    else:
                        sq_dist = np.linalg.norm(xj - xi) ** 2
                    M2[i, j] = sq_dist

            if sinkhorn:
                d_emd2 = ot.sinkhorn2(a, b, M2, 1e-3)
            else:
                d_emd2 = ot.emd2(a, b, M2)

            w2_dists[i_n, i_logvarx_true] = d_emd2
    return w2_dists
