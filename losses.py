"""Losses."""

import torch
import torch.nn
from torch.nn import functional as F


def bce_on_intensities(x, recon_x, scale_b):
    """BCE summed over the voxels intensities."""
    width = recon_x.shape[0]
    height = recon_x.shape[1]
    bce = torch.sum(
        F.binary_cross_entropy(recon_x, x) / scale_b.exp() + 2 * scale_b)
    return bce


def kullback_leibler(mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def vae_loss(x, recon_x, scale_b, mu, logvar):
    bce = bce_on_intensities(x, recon_x, scale_b)
    kld = kullback_leibler(mu, logvar)
    return bce + kld
