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
    # print('BCE: %s KLD: %s' % (bce.item(), kld.item()))
    return bce + kld


def gan_loss(predicted_labels, true_labels):
    #print('predict shape = (%d, %d)' % predicted_labels.shape)
    #print('true shape = (%d)' % true_labels.shape)
    bce = F.binary_cross_entropy(predicted_labels, true_labels)
    return bce


def regularization_adversarial(discriminator,
                               real_recon_batch,
                               fake_recon_batch):
    #print('real_recon shape (%d, %d, %d, %d)' % real_recon_batch.shape)
    #print('fake_recon shape (%d, %d, %d, %d)' % fake_recon_batch.shape)
    batch_size = real_recon_batch.shape[0]
    real_labels = torch.full(
        (batch_size,), REAL_LABEL, device=DEVICE)
    fake_labels = torch.full(
        (batch_size,), FAKE_LABEL, device=DEVICE)

    # discriminator - real
    predicted_labels_real = discriminator(real_recon_batch)
    loss_real = gan_loss(
        predicted_labels=predicted_labels_real,
        true_labels=real_labels)

    # discriminator - fake
    predicted_labels_fake = discriminator(fake_recon_batch)
    loss_fake = gan_loss(
        predicted_labels=predicted_labels_fake,
        true_labels=fake_labels)

    loss_discriminator = loss_real + loss_fake

    # generator/decoder - wants to fool the discriminator
    loss_generator = gan_loss(
        predicted_labels=predicted_labels_fake,
        true_labels=real_labels)

    loss_regularization = loss_discriminator + loss_generator
    return loss_regularization
