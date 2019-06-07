"""Utils for getting datasets."""

import glob
import logging
import os
import pickle

import numpy as np
import torch
import torch.utils
from torchvision import datasets, transforms
import skimage

CUDA = torch.cuda.is_available()
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

IMG_SHAPE = (28, 28)

TRAIN_VAL_DATASETS = '/neuro/train_val_datasets'


def get_loaders(dataset_name, frac_val, batch_size,
                img_shape=IMG_SHAPE, kwargs=KWARGS):
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
    elif dataset_name == 'cryo':
        dataset = get_dataset_cryo(frac_val, batch_size, img_shape, kwargs)
    elif dataset_name in ['mri', 'segmentation', 'fmri']:
        train_loader, val_loader = get_loaders_brain(
            dataset_name, frac_val, batch_size, img_shape, kwargs)
        return train_loader, val_loader
    else:
        raise ValueError('Unknown dataset name: %s' % dataset_name)
    length = len(dataset)
    train_length = int((1 - frac_val) * length)
    val_length = int(length - train_length)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length])

    if dataset_name == 'mnist' or dataset_name == 'cryo':
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


def get_dataset_cryo(frac_val, batch_size, img_shape=IMG_SHAPE, kwargs=KWARGS):
    paths = glob.glob('/cryo/job40_vs_job034/*.pkl')
    all_datasets = []
    for path in paths:
        with open(path, 'rb') as pkl:
            data = pickle.load(pkl)
            dataset = data['ParticleStack']
            img_h, img_w = IMG_SHAPE
            dataset = skimage.transform.resize(
                dataset, (len(dataset), img_h, img_w))
            all_datasets.append(dataset)
    all_datasets = np.vstack([d for d in all_datasets])
    dataset = torch.Tensor(all_datasets)
    return dataset


def get_loaders_brain(dataset_name, frac_val, batch_size,
                      img_shape=IMG_SHAPE, kwargs=KWARGS):
    if len(img_shape) == 2:
        shape_str = '%dx%d' % img_shape
    elif len(img_shape) == 3:
        shape_str = '%dx%dx%d' % img_shape
    else:
        raise ValueError('Weird image shape.')
    train_path = os.path.join(
        TRAIN_VAL_DATASETS, 'train_%s_%s.npy' % (dataset_name, shape_str))
    val_path = os.path.join(
        TRAIN_VAL_DATASETS, 'val_%s_%s.npy' % (dataset_name, shape_str))

    train = torch.Tensor(np.load(train_path))
    val = torch.Tensor(np.load(val_path))
    logging.info('-- Train tensor: (%d, %d, %d, %d)' % train.shape)
    np.random.shuffle(train)
    train_dataset = torch.utils.data.TensorDataset(train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    logging.info('-- Val tensor: (%d, %d, %d, %d)' % val.shape)
    val_dataset = torch.utils.data.TensorDataset(val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader
