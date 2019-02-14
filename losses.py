"""Losses."""

import torch
import torch.nn
from torch.nn import functional as F


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

REAL_LABEL = 1
FAKE_LABEL = 0


def bce_on_intensities(x, recon_x, scale_b):
    """BCE summed over the voxels intensities."""
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


def adversarial(discriminator, real_recon_batch, fake_recon_batch):
    batch_size = real_recon_batch.shape[0]
    fake_batch_size = fake_recon_batch.shape[0]
    assert batch_size == fake_batch_size
    real_labels = torch.full(
        (batch_size,), REAL_LABEL, device=DEVICE)
    fake_labels = torch.full(
        (batch_size,), FAKE_LABEL, device=DEVICE)

    # discriminator - real
    predicted_labels_real = discriminator(real_recon_batch)
    loss_real = F.binary_cross_entropy(
        predicted_labels_real,
        real_labels)

    # discriminator - fake
    predicted_labels_fake = discriminator(fake_recon_batch)
    loss_fake = F.binary_cross_entropy(
        predicted_labels_fake,
        fake_labels)

    loss_discriminator = loss_real + loss_fake

    # generator/decoder - wants to fool the discriminator
    loss_generator = F.binary_cross_entropy(
        predicted_labels_fake,
        real_labels)

    loss_regularization = loss_discriminator + loss_generator
    return loss_regularization
