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


def iw_vae_loss(x, recon_x, mu, logvar, z):
    var = torch.exp(logvar)
    log_QzGx = torch.sum(- 0.5 * (z - mu) ** 2 / var - 0.5 * logvar, -1)

    log_Pz = torch.sum(-0.5 * z ** 2, -1)

    # Note: reconstruction is a cross-entropy here
    log_PxGz = torch.sum(
        x * torch.log(recon_x) + (1 - x) * torch.log(1 - recon_x), -1)

    log_weight = log_Pz + log_PxGz - log_QzGx
    log_weight = log_weight - torch.max(log_weight, 0)[0]
    weight = torch.exp(log_weight)
    weight = weight / torch.sum(weight, 0)
    weight = torch.Variable(weight.data, requires_grad=False)
    loss = -torch.mean(torch.sum(weight * (log_Pz + log_PxGz - log_QzGx), 0))
    return loss
