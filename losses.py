"""Losses."""

import torch
import torch.nn
from torch.nn import functional as F


# TODO(nina): Average on intensities, instead of sum.

def bce_on_intensities(x, recon_x, scale_b):
    """
    BCE summed over the voxels intensities.
    scale_b: plays role of loss' weighting factor.
    """
    bce = torch.sum(
        F.binary_cross_entropy(recon_x, x) / scale_b.exp() + 2 * scale_b)
    return bce


def mse_on_intensities(x, recon_x, scale_b):
    """
    MSE summed over the voxels intensities.
    scale_b: plays role of loss' weighting factor.
    """
    mse = F.mse_loss(recon_x, x, reduction='sum') / scale_b
    return mse


def mse_on_features(feature, recon_feature, logvar):
    """
    MSE over features of FC layer of Discriminator.
    sigma2: plays role of loss' weighting factor.
    """
    mse = F.mse_loss(recon_feature, feature) / (2 * logvar.exp())
    mse = torch.mean(mse)
    return mse


def kullback_leibler(mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def vae_loss(x, recon_x, scale_b, mu, logvar):
    bce = bce_on_intensities(x, recon_x, scale_b)
    kld = kullback_leibler(mu, logvar)
    return bce + kld
