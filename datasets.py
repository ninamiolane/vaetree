"""Utils for getting datasets."""

import csv
import glob
import gzip
import h5py
import logging
import os
import pandas as pd
import pickle

import numpy as np
import torch
import torch.utils
import skimage

from geomstats.spd_matrices_space import SPDMatricesSpace
from torchvision import datasets, transforms
from urllib import request

CUDA = torch.cuda.is_available()
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

CRYO_DIR = '/cryo'
NEURO_DIR = '/neuro'

NEURO_TRAIN_VAL_DIR = os.path.join(NEURO_DIR, 'train_val_datasets')
CRYO_TRAIN_VAL_DIR = os.path.join(CRYO_DIR, 'train_val_datasets')

N_NODES = 28
CORR_THRESH = 0.1
GAMMA = 1.0
N_GRAPHS = 86
ID_COEF = 4  # Make Positive definite

FRAC_VAL = 0.05

# TODO(nina): Reorganize:
# get_datasets provide train/val in np.array,
# get_loaders shuflles and transforms in tensors/loaders


def get_datasets(dataset_name, frac_val=FRAC_VAL, batch_size=8,
                 img_shape=None, kwargs=KWARGS):

    img_shape_no_channel = None
    if img_shape is not None:
        img_shape_no_channel = img_shape[1:]
    # TODO(nina): Consistency in datasets: add channels for all
    logging.info('Loading data from dataset: %s' % dataset_name)
    if dataset_name == 'mnist':
        train_dataset, val_dataset = get_dataset_mnist()
    elif dataset_name == 'omniglot':
        if img_shape_no_channel is not None:
            transform = transforms.Compose([
                transforms.Resize(img_shape_no_channel),
                transforms.ToTensor()])
        else:
            transform = transforms.ToTensor()
        dataset = datasets.Omniglot(
            '../data', download=True, transform=transform)
        train_dataset, val_dataset = split_dataset(
            dataset, frac_val=frac_val)
    elif dataset_name in [
            'randomrot1D_nodisorder',
            'randomrot1D_multiPDB',
            'randomrot_nodisorder']:
        dataset = get_dataset_cryo(dataset_name, img_shape_no_channel, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'cryo_sphere':
        dataset = get_dataset_cryo_sphere(img_shape_no_channel, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'cryo_exp':
        dataset = get_dataset_cryo_exp(img_shape_no_channel, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'connectomes':
        train_dataset, val_dataset = get_dataset_connectomes(
            img_shape_no_channel=img_shape_no_channel)
    elif dataset_name == 'connectomes_simu':
        train_dataset, val_dataset = get_dataset_connectomes_simu(
            img_shape_no_channel=img_shape_no_channel)
    elif dataset_name == 'connectomes_schizophrenia':
        train_dataset, val_dataset, _ = get_dataset_connectomes_schizophrenia()
    elif dataset_name in ['mri', 'segmentation', 'fmri']:
        train_loader, val_loader = get_loaders_brain(
            dataset_name, frac_val, batch_size, img_shape_no_channel, kwargs)
        return train_loader, val_loader
    else:
        raise ValueError('Unknown dataset name: %s' % dataset_name)

    return train_dataset, val_dataset


def split_dataset(dataset, frac_val=FRAC_VAL):
    length = len(dataset)
    train_length = int((1 - frac_val) * length)
    train_dataset = dataset[:train_length]
    val_dataset = dataset[train_length:]
    return train_dataset, val_dataset


def get_shape_string(img_shape_no_channel):
    if len(img_shape_no_channel) == 2:
        shape_str = '%dx%d' % img_shape_no_channel
    elif len(img_shape_no_channel) == 3:
        shape_str = '%dx%dx%d' % img_shape_no_channel
    else:
        raise ValueError('Weird image shape.')
    return shape_str


def normalization_linear(dataset):
    for i in range(len(dataset)):
        data = dataset[i]
        min_data = np.min(data)
        max_data = np.max(data)
        dataset[i] = (data - min_data) / (max_data - min_data)
    return dataset


def add_channels(dataset, img_dim=2):
    if dataset.ndim == 3:
        dataset = np.expand_dims(dataset, axis=1)
    return dataset


def is_pos_def(x):
    eig, _ = np.linalg.eig(x)
    return (eig > 0).all()


def is_sym(x):
    return np.all(np.isclose(x, np.transpose(x), rtol=1e-4))


def is_spd(x):
    """Assumes the matrix is symmetric"""
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    elif x.ndim == 4:
        x = x[:, 0, :, :]
    _, n, _ = x.shape
    all_spd = True
    for i, one_mat in enumerate(x):
        if not is_pos_def(one_mat):
            print('problem pos def at %d' % i)
            print(np.linalg.eig(one_mat)[0])
        if not is_sym(one_mat):
            print('problem sym at %d' % i)
            print(one_mat - np.transpose(one_mat))

        all_spd = all_spd & is_sym(one_mat) & is_pos_def(one_mat)
    return all_spd


def r_pearson_from_z_score(mat):
    """Inverse Fisher transformation"""
    r_mat = np.tanh(mat)
    return r_mat


def get_dataset_mnist(img_shape_no_channel=(28, 28)):
    shape_str = get_shape_string(img_shape_no_channel)
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_mnist_%s.npy' % shape_str)
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_mnist_%s.npy' % shape_str)

    train_exists = os.path.isfile(train_path)
    val_exists = os.path.isfile(val_path)
    if train_exists and val_exists:
        print('Loading %s...' % train_path)
        print('Loading %s...' % val_path)
        train_dataset = np.load(train_path)
        val_dataset = np.load(val_path)
    else:
        filename = [
            ['training_images', 'train-images-idx3-ubyte.gz'],
            ['test_images', 't10k-images-idx3-ubyte.gz'],
            ['training_labels', 'train-labels-idx1-ubyte.gz'],
            ['test_labels', 't10k-labels-idx1-ubyte.gz']
            ]
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        save_folder = '/neuro/train_val_datasets/'
        for name in filename:
            print('Downloading ' + name[1] + '...')
            request.urlretrieve(base_url + name[1], save_folder + name[1])
        print('Download complete.')

        mnist = {}

        for name in filename[:2]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(
                    f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        for name in filename[-2:]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open('mnist.pkl', 'wb') as f:
            pickle.dump(mnist, f)
        print('Save complete.')

        with open('mnist.pkl', 'rb') as f:
            mnist = pickle.load(f)  # training_labels, test_labels also

        dataset = mnist['training_images']
        dataset = add_channels(dataset, img_dim=2)
        dataset = dataset / 255  # normalization
        train_dataset, val_dataset = split_dataset(
                dataset, frac_val=FRAC_VAL)
        print('Saving %s...' % train_path)
        print('Saving %s...' % val_path)
        np.save(train_path, train_dataset)
        np.save(val_path, val_dataset)

    return train_dataset, val_dataset


def get_dataset_connectomes(img_shape_no_channel=(100, 100),
                            partial_corr=True):
    """
    Connectomes from HCP 1200:
    https://www.humanconnectome.org/storage/app/media/
    documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf
    """
    netmat_type = int(partial_corr) + 1

    shape_str = get_shape_string(img_shape_no_channel)
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_conn_%s.npy' % shape_str)
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_conn_%s.npy' % shape_str)

    train_exists = os.path.isfile(train_path)
    val_exists = os.path.isfile(val_path)
    if train_exists and val_exists:
        print('Loading %s...' % train_path)
        print('Loading %s...' % val_path)
        train_dataset = np.load(train_path)
        val_dataset = np.load(val_path)

    else:
        n_nodes = img_shape_no_channel[0]
        hcp_dir = os.path.join(
            NEURO_DIR, 'HCP_PTN1200_recon2')
        netmats_path = os.path.join(
            hcp_dir, 'netmats/3T_HCP1200_MSMAll_d%d_ts2/netmats%d.txt' % (
                n_nodes, netmat_type))
        print('Loading %s...' % netmats_path)
        netmats = np.loadtxt(netmats_path)
        netmats = netmats.reshape(-1, n_nodes, n_nodes)
        n_mats, _, _ = netmats.shape

        r_mats = r_pearson_from_z_score(netmats)

        # HACK
        r_mats = 1 / 4 * (r_mats + ID_COEF * np.tile(
            np.eye(n_nodes, n_nodes), (n_mats, 1, 1)))
        r_mats = np.abs(r_mats)

        r_mats = add_channels(r_mats, img_dim=2)

        dataset = r_mats

        assert len(dataset.shape) == 4

        # dataset = normalization_linear(dataset)

        train_dataset, val_dataset = split_dataset(dataset)
        print('Saving %s...' % train_path)
        print('Saving %s...' % val_path)
        np.save(train_path, train_dataset)
        np.save(val_path, val_dataset)

    return train_dataset, val_dataset


def get_dataset_connectomes_simu(img_shape_no_channel=(15, 15)):
    """
    Simulating a geodesic triangle.
    """
    shape_str = get_shape_string(img_shape_no_channel)
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_conn_simu_%s.npy' % shape_str)
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_conn_simu_%s.npy' % shape_str)

    train_exists = os.path.isfile(train_path)
    val_exists = os.path.isfile(val_path)
    if train_exists and val_exists:
        print('Loading %s...' % train_path)
        print('Loading %s...' % val_path)
        train_dataset = np.load(train_path)
        val_dataset = np.load(val_path)

    else:
        n, _ = img_shape_no_channel
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        spd_space = SPDMatricesSpace(n=n)
        vec_dim = int(n * (n + 1) / 2)
        vec_a = np.zeros(vec_dim)
        vec_b = np.zeros(vec_dim)
        vec_c = np.zeros(vec_dim)

        cos_angle = np.cos(np.pi / 3)
        sin_angle = np.cos(np.pi / 3)
        vec_a[0] = cos_angle
        vec_a[1] = sin_angle
        vec_b[0] = -cos_angle
        vec_c[1] = sin_angle
        vec_c[0] = 0.
        vec_c[1] = -1.

        mat_identity = np.eye(n)
        sym_mat_a = spd_space.symmetric_matrix_from_vector(vec_a)
        spd_mat_a = spd_space.metric.exp(
            base_point=mat_identity, tangent_vec=sym_mat_a)
        sym_mat_b = spd_space.symmetric_matrix_from_vector(vec_b)
        spd_mat_b = spd_space.metric.exp(
            base_point=mat_identity, tangent_vec=sym_mat_b)
        sym_mat_c = spd_space.symmetric_matrix_from_vector(vec_c)
        spd_mat_c = spd_space.metric.exp(
            base_point=mat_identity, tangent_vec=sym_mat_c)

        assert is_spd(spd_mat_a)
        assert is_spd(spd_mat_b)
        assert is_spd(spd_mat_c)

        vec_ab = spd_space.metric.log(base_point=spd_mat_a, point=spd_mat_b)
        geod_ab = spd_space.metric.geodesic(
            initial_point=spd_mat_a, initial_tangent_vec=vec_ab)
        points_ab = geod_ab(np.arange(0, 1, 0.0002))
        assert is_spd(points_ab)

        vec_bc = spd_space.metric.log(base_point=spd_mat_b, point=spd_mat_c)
        geod_bc = spd_space.metric.geodesic(
            initial_point=spd_mat_b, initial_tangent_vec=vec_bc)
        points_bc = geod_bc(np.arange(0, 1, 0.0002))
        assert is_spd(points_bc)

        vec_ca = spd_space.metric.log(base_point=spd_mat_c, point=spd_mat_a)
        geod_ca = spd_space.metric.geodesic(
            initial_point=spd_mat_c, initial_tangent_vec=vec_ca)
        points_ca = geod_ca(np.arange(0, 1, 0.0002))
        assert is_spd(points_ca)
        os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

        dataset = np.concatenate([points_ab, points_bc, points_ca], axis=0)
        assert is_spd(dataset)
        assert np.all(spd_space.belongs(dataset))
        np.random.shuffle(dataset)

        dataset = add_channels(dataset, img_dim=2)
        assert len(dataset.shape) == 4

        # dataset = normalization_linear(dataset)

        train_dataset, val_dataset = split_dataset(dataset)
        print('Saving %s...' % train_path)
        print('Saving %s...' % val_path)
        np.save(train_path, train_dataset)
        np.save(val_path, val_dataset)

    return train_dataset, val_dataset


def get_dataset_connectomes_schizophrenia():
    """
    Connectomes are SPD matrices of size N_NODESxN_NODES.
    """
    graphs = pd.read_csv('/neuro/connectomes/train_fnc.csv')
    map_functional = pd.read_csv(
        '/neuro/connectomes/comp_ind_fmri.csv', index_col=None)
    map_functional = map_functional['fMRI_comp_ind'].to_dict()
    map_functional_r = {v: k for k, v
                        in map_functional.items()}
    mapping = pd.read_csv(
        '/neuro/connectomes/rs_fmri_fnc_mapping.csv')
    graph_labels = pd.read_csv('/neuro/connectomes/train_labels.csv')
    all_graphs = [None] * N_GRAPHS
    all_labels = np.zeros(N_GRAPHS)

    def create_connectome(graph_id, mapping):
        u = np.zeros((N_NODES, N_NODES))
        nb_edges = mapping.shape[0]
        for edge in range(nb_edges):
            e0, e1 = (mapping.iloc[edge]['mapA'], mapping.iloc[edge]['mapB'])
            region0, region1 = map_functional_r[e0], map_functional_r[e1]
            corr = graphs.iloc[graph_id][edge+1]
            u[region0, region1] = corr
        u = np.multiply(u, (np.abs(u) > CORR_THRESH))
        return np.abs(u + u.T)

    for graph_id in range(N_GRAPHS):
        all_graphs[graph_id] = create_connectome(graph_id, mapping)
        all_labels[graph_id] = int(
            graph_labels.loc[graphs.index[graph_id], 'Class'])

    all_labels = np.array(all_labels)
    all_graphs = np.array(all_graphs)
    train_dataset, val_dataset = split_dataset(all_graphs)
    train_dataset = torch.Tensor(train_dataset)
    val_dataset = torch.Tensor(val_dataset)

    return train_dataset, val_dataset, all_labels


def get_dataset_cryo_sphere(img_shape_no_channel=None, kwargs=KWARGS):
    shape_str = get_shape_string(img_shape_no_channel)
    cryo_path = os.path.join(
        CRYO_TRAIN_VAL_DIR, 'cryo_%s.npy' % shape_str)

    if os.path.isfile(cryo_path):
        all_datasets = np.load(cryo_path)
    else:
        paths = glob.glob('/cryo/job40_vs_job034/*.pkl')
        all_datasets = []
        for path in paths:
            with open(path, 'rb') as pkl:
                logging.info('Loading file %s...' % path)
                data = pickle.load(pkl)
                dataset = data['ParticleStack']
                n_data = len(dataset)
                if img_shape_no_channel is not None:
                    img_h, img_w = img_shape_no_channel
                    dataset = skimage.transform.resize(
                        dataset, (n_data, img_h, img_w))

                dataset = normalization_linear(dataset)

                all_datasets.append(dataset)
        all_datasets = np.vstack([d for d in all_datasets])
        all_datasets = np.expand_dims(all_datasets, axis=1)
        np.save(cryo_path, all_datasets)

    logging.info('Cryo dataset: (%d, %d, %d, %d)' % all_datasets.shape)
    dataset = torch.Tensor(all_datasets)
    return dataset


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]  # item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans


def get_dataset_cryo(
        dataset_name, img_shape_no_channel=None, kwargs=KWARGS):
    shape_str = get_shape_string(img_shape_no_channel)
    cryo_img_path = os.path.join(
        CRYO_TRAIN_VAL_DIR,
        'cryo_%s_%s.npy' % (dataset_name, shape_str))
    cryo_labels_path = os.path.join(
        CRYO_TRAIN_VAL_DIR,
        'cryo_labels_%s_%s.csv' % (dataset_name, shape_str))

    if os.path.isfile(cryo_img_path) and os.path.isfile(cryo_labels_path):
        all_datasets = np.load(cryo_img_path)

    else:
        if not os.path.isdir('/cryo/%s/' % dataset_name):
            os.system("cd /cryo/")
            os.system("source osf_dl_folder %s" % dataset_name)
        paths = glob.glob('/cryo/%s/final/*.h5' % dataset_name)
        all_datasets = []
        all_focuses = []
        all_thetas = []
        for path in paths:
            logging.info('Loading file %s...' % path)
            data_dict = load_dict_from_hdf5(path)
            dataset = data_dict['data']
            n_data = len(dataset)

            focus = data_dict['optics']['defocus_nominal']
            focus = np.repeat(focus, n_data)
            theta = data_dict['coordinates'][:, 3]

            if img_shape_no_channel is not None:
                img_h, img_w = img_shape_no_channel
                dataset = skimage.transform.resize(
                    dataset, (n_data, img_h, img_w))
            dataset = normalization_linear(dataset)

            all_datasets.append(dataset)
            all_focuses.append(focus)
            all_thetas.append(theta)

        all_datasets = np.vstack([d for d in all_datasets])
        all_datasets = np.expand_dims(all_datasets, axis=1)

        all_focuses = np.concatenate(all_focuses, axis=0)
        all_focuses = np.expand_dims(all_focuses, axis=1)
        all_thetas = np.concatenate(all_thetas, axis=0)
        all_thetas = np.expand_dims(all_thetas, axis=1)

        assert len(all_datasets) == len(all_focuses)
        assert len(all_datasets) == len(all_thetas)

        np.save(cryo_img_path, all_datasets)
        with open(cryo_labels_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['focus', 'theta'])
            for focus, theta in zip(all_focuses, all_thetas):
                writer.writerow([focus[0], theta[0]])
    dataset = torch.Tensor(all_datasets)
    return dataset


def get_dataset_cryo_exp(img_shape_no_channel=None, kwargs=KWARGS):
    NEURO_TRAIN_VAL_DIR = os.path.join(CRYO_DIR, 'train_val_datasets')
    shape_str = get_shape_string(img_shape_no_channel)
    cryo_img_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'cryo_exp_%s.npy' % shape_str)
    if os.path.isfile(cryo_img_path):
        dataset = np.load(cryo_img_path)

    else:
        path = os.path.join(CRYO_DIR, 'particles.h5')

        logging.info('Loading file %s...' % path)
        data_dict = load_dict_from_hdf5(path)
        dataset = data_dict['particles']

        if img_shape_no_channel is not None:
            n_data = len(dataset)
            img_h, img_w = img_shape_no_channel
            dataset = skimage.transform.resize(
                dataset, (n_data, img_h, img_w))

        dataset = normalization_linear(dataset)
        dataset = np.expand_dims(dataset, axis=1)

        np.save(cryo_img_path, dataset)

    dataset = torch.Tensor(dataset)
    return dataset


def get_loaders_brain(dataset_name, frac_val, batch_size,
                      img_shape, kwargs=KWARGS):

    shape_str = get_shape_string(img_shape)
    NEURO_TRAIN_VAL_DIR = os.path.join(NEURO_DIR, 'train_val_datasets')
    train_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'train_%s_%s.npy' % (dataset_name, shape_str))
    val_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'val_%s_%s.npy' % (dataset_name, shape_str))

    train = torch.Tensor(np.load(train_path))
    val = torch.Tensor(np.load(val_path))

    logging.info('-- Train tensor: (%d, %d, %d, %d)' % train.shape)
    train_dataset = torch.utils.data.TensorDataset(train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    logging.info('-- Val tensor: (%d, %d, %d, %d)' % val.shape)
    val_dataset = torch.utils.data.TensorDataset(val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader
