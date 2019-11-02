"""Geometric constants and tools"""


import importlib
import numpy as np
import os

import torch

import geomstats
from geomstats.geometry.euclidean_space import EuclideanSpace
from geomstats.geometry.hyperbolic_space import HyperbolicSpace
from geomstats.geometry.hypersphere import Hypersphere

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


R2 = EuclideanSpace(dimension=2)
R3 = EuclideanSpace(dimension=3)
H2 = HyperbolicSpace(dimension=2)
S2 = Hypersphere(dimension=2)
MANIFOLD = {'r2': R2, 's2': S2, 'h2': H2}


def manifold_and_base_point(manifold_name):
    manifold = MANIFOLD[manifold_name]
    if os.environ['GEOMSTATS_BACKEND'] == 'numpy':
        if manifold_name == 's2':
            base_point = np.array([0, 0, 1])
        elif manifold_name == 'h2':
            base_point = np.array([1, 0, 0])
        elif manifold_name == 'r2':
            base_point = np.array([0, 0])
        else:
            raise ValueError('Manifold not supported.')
    elif os.environ['GEOMSTATS_BACKEND'] == 'pytorch':
        if manifold_name == 's2':
            base_point = torch.Tensor([0., 0., 1.]).to(DEVICE)
        elif manifold_name == 'h2':
            base_point = torch.Tensor([1., 0., 0.]).to(DEVICE)
        elif manifold_name == 'r2':
            base_point = torch.Tensor([0., 0.]).to(DEVICE)
        else:
            raise ValueError('Manifold not supported.')
    return manifold, base_point


def convert_to_tangent_space(x, manifold_name='s2'):
    n_samples, _ = x.shape
    if type(x) == np.ndarray:
        if manifold_name == 's2':
            x_vector_extrinsic = np.hstack([x, np.zeros((n_samples, 1))])
        elif manifold_name == 'h2':
            x_vector_extrinsic = np.hstack([np.zeros((n_samples, 1)), x])
        elif manifold_name == 'r2':
            x_vector_extrinsic = x
        else:
            raise ValueError('Manifold not supported.')
    elif type(x) == torch.Tensor:
        if os.environ['GEOMSTATS_BACKEND'] == 'numpy':
            os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
            importlib.reload(geomstats.backend)
        if manifold_name == 's2':
            x_vector_extrinsic = torch.cat(
                [x, torch.zeros((n_samples, 1)).to(DEVICE)], dim=1)
        elif manifold_name == 'h2':
            x_vector_extrinsic = torch.cat(
                [torch.zeros((n_samples, 1)).to(DEVICE), x], dim=1)
        elif manifold_name == 'r2':
            x_vector_extrinsic = x
        else:
            raise ValueError('Manifold not supported.')

    return x_vector_extrinsic
