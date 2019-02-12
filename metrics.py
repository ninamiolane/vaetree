"""Metrics to evaluate performances."""

import numpy as np
from scipy import linalg
import torch
from torch.nn import functional as F


def binary_cross_entropy(input, target):
    """
    BCE averaged over the voxels of the images.
    """
    bce = F.binary_cross_entropy(input, target, reduction='mean')
    return bce


def mse(input, target):
    """
    MSE averaged over the voxels of the images.
    """
    mse = F.mse_loss(input, target, reduction='mean')
    return mse


def l1_norm(input, target):
    """
    L1 norm averaged over the voxels of the images.
    """
    l1_norm = torch.mean(torch.abs(target - input))
    return l1_norm


def mutual_information(input, target):
    """
    Mutual information for joint histogram of two images.
    https://en.wikipedia.org/wiki/Mutual_information
    """

    hist_2d, x_edges, y_edges = np.histogram2d(
        target.ravel(),
        input.ravel(),
        bins=100)

    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def frechet_distance_gaussians(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    The Frechet distance between two multivariate Gaussians
    * X_1 ~ N(mu_1, C_1)
    * X_2 ~ N(mu_2, C_2)
    is
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def frechet_inception_distance(targets, inputs):
    """
    FID between two datasets of images.
    FID is meant to be more consistent than Inception Score.

    From: https://github.com/mseitzer/pytorch-fid
    """

    targets = targets.ravel()
    inputs = inputs.ravel()

    mu_targets = np.mean(targets, axis=0)
    sigma_targets = np.cov(targets, rowvar=False)

    mu_inputs = np.mean(inputs, axis=0)
    sigma_inputs = np.cov(inputs, rowvar=False)

    fid = frechet_distance_gaussians(
        mu_targets, sigma_targets, mu_inputs, sigma_inputs)
    return fid
