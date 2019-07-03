"""Utils for getting datasets."""

import csv
import glob
import h5py
import logging
import os
import pandas as pd
import pickle

import numpy as np
import torch
import torch.utils
import skimage

from torchvision import datasets, transforms

CUDA = torch.cuda.is_available()
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

IMG_SHAPE = (128, 128)

CRYO_DIR = '/cryo'
NEURO_DIR = '/neuro'

NEURO_TRAIN_VAL_DIR = '/neuro/train_val_datasets'
CRYO_TRAIN_VAL_DIR = '/cryo/train_val_datasets'

N_NODES = 28
CORR_THRESH = 0.1
GAMMA = 1.0
N_GRAPHS = 86

FRAC_VAL = 0.2

# TODO(nina): Reorganize:
# get_datasets provide train/val in np.array,
# get_loaders shuflles and transforms in tensors/loaders


def get_loaders(dataset_name, frac_val=FRAC_VAL, batch_size=8,
                img_shape=IMG_SHAPE, kwargs=KWARGS):
    # TODO(nina): Consistency in datasets: add channels for all
    logging.info('Loading data from dataset: %s' % dataset_name)
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.ToTensor())
    elif dataset_name == 'omniglot':
        dataset = datasets.Omniglot(
            '../data', download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_shape), transforms.ToTensor()]))
    elif dataset_name in [
            'randomrot1D_nodisorder',
            'randomrot1D_multiPDB',
            'randomrot_nodisorder']:
        dataset = get_dataset_cryo(dataset_name, img_shape, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'cryo_sphere':
        dataset = get_dataset_cryo_sphere(img_shape, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'cryo_exp':
        dataset = get_dataset_cryo_exp(img_shape, kwargs)
        train_dataset, val_dataset = split_dataset(dataset)
    elif dataset_name == 'connectomes':
        train_dataset, val_dataset = get_dataset_connectomes(
            img_shape=img_shape)
    elif dataset_name == 'connectomes_schizophrenia':
        dataset, _ = get_dataset_connectomes_schizophrenia()
    elif dataset_name in ['mri', 'segmentation', 'fmri']:
        train_loader, val_loader = get_loaders_brain(
            dataset_name, frac_val, batch_size, img_shape, kwargs)
        return train_loader, val_loader
    else:
        raise ValueError('Unknown dataset name: %s' % dataset_name)

    train_dataset = torch.Tensor(train_dataset)
    val_dataset = torch.Tensor(val_dataset)
    if dataset_name in ['mnist']:
        train_tensor = train_dataset.dataset.data[train_dataset.indices]
        val_tensor = val_dataset.dataset.data[val_dataset.indices]
        logging.info(
            '-- Train tensor: (%d, %d, %d)' % train_tensor.shape)
        logging.info(
            '-- Valid tensor: (%d, %d, %d)' % val_tensor.shape)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader


def split_dataset(dataset, frac_val=FRAC_VAL):
    length = len(dataset)
    train_length = int((1 - frac_val) * length)
    train_dataset = dataset[:train_length]
    val_dataset = dataset[train_length:]
    return train_dataset, val_dataset


def get_shape_string(img_shape=IMG_SHAPE):
    if len(img_shape) == 2:
        shape_str = '%dx%d' % img_shape
    elif len(img_shape) == 3:
        shape_str = '%dx%dx%d' % img_shape
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


def get_dataset_connectomes(img_shape=(100, 100), partial_corr=True):
    """
    Connectomes from HCP 1200:
    https://www.humanconnectome.org/storage/app/media/
    documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf
    """
    netmat_type = int(partial_corr) + 1

    shape_str = get_shape_string(img_shape)
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
        n_nodes = img_shape[0]
        hcp_dir = os.path.join(
            NEURO_DIR, 'HCP_PTN1200_recon2')
        netmats_path = os.path.join(
            hcp_dir, 'netmats/3T_HCP1200_MSMAll_d%d_ts2/netmats%d.txt' % (
                n_nodes, netmat_type))
        print('Loading %s...' % netmats_path)
        netmats = np.loadtxt(netmats_path)
        netmats = netmats.reshape(-1, n_nodes, n_nodes)

        netmats = np.expand_dims(netmats, axis=1)
        netmats = normalization_linear(netmats)
        dataset = netmats

        assert len(dataset.shape) == 4

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
    return all_graphs, all_labels


def get_dataset_cryo_sphere(img_shape=IMG_SHAPE, kwargs=KWARGS):
    shape_str = get_shape_string(img_shape)
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
                img_h, img_w = img_shape
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
        dataset_name, img_shape=IMG_SHAPE, kwargs=KWARGS):
    shape_str = get_shape_string(img_shape)
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

            img_h, img_w = img_shape
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


def get_dataset_cryo_exp(img_shape=IMG_SHAPE, kwargs=KWARGS):
    NEURO_TRAIN_VAL_DIR = os.path.join(CRYO_DIR, 'train_val_datasets')
    shape_str = get_shape_string(img_shape)
    cryo_img_path = os.path.join(
        NEURO_TRAIN_VAL_DIR, 'cryo_exp_%s.npy' % shape_str)
    if os.path.isfile(cryo_img_path):
        dataset = np.load(cryo_img_path)

    else:
        path = os.path.join(CRYO_DIR, 'particles.h5')

        logging.info('Loading file %s...' % path)
        data_dict = load_dict_from_hdf5(path)
        dataset = data_dict['particles']
        n_data = len(dataset)

        img_h, img_w = img_shape
        dataset = skimage.transform.resize(
            dataset, (n_data, img_h, img_w))
        dataset = normalization_linear(dataset)
        dataset = np.expand_dims(dataset, axis=1)

        np.save(cryo_img_path, dataset)

    dataset = torch.Tensor(dataset)
    return dataset


def get_loaders_brain(dataset_name, frac_val, batch_size,
                      img_shape=IMG_SHAPE, kwargs=KWARGS):

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
