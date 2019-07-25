"""Toy losses."""

import math
import numpy as np
import torch
import torch.nn
from torch.nn import functional as F

import toynn

from geomstats.spd_matrices_space import SPDMatricesSpace
from geomstats.general_linear_group import GeneralLinearGroup


CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')


def is_pos_def(x):
    eig, _ = torch.symeig(x, eigenvectors=True)
    return (eig > 0).all()


def is_spd(x):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    _, n, _ = x.shape
    gln_group = GeneralLinearGroup(n=n)
    for one_mat in x:
        assert is_pos_def(one_mat)
        assert gln_group.belongs(one_mat)


def riem_square_distance(batch_data, batch_recon):
    n_batch_data, dim = batch_data.shape
    n = int(np.sqrt(dim))
    spd_space = SPDMatricesSpace(n=n)
    batch_data_mat = batch_data.resize(n_batch_data, n, n)
    batch_recon_mat = batch_recon.resize(n_batch_data, n, n)

    is_spd(batch_data_mat)
    # is_spd(batch_recon_mat)

    sq_dist = spd_space.metric.squared_dist(
        batch_data_mat, batch_recon_mat)
    return sq_dist


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def fa_neg_loglikelihood(weight, data):
    weight = weight.cpu()
    sig2 = torch.mean(data ** 2, dim=0)

    loglikelihood_term_1 = - 1. / 2. * torch.log(
        2 * np.pi * (weight ** 2 + 1))
    loglikelihood_term_2 = - sig2 / (2 * (weight ** 2 + 1))
    loglikelihood = loglikelihood_term_1 + loglikelihood_term_2
    neg_loglikelihood = - loglikelihood
    return neg_loglikelihood


def reconstruction_loss(batch_data, batch_recon, batch_logvarx,
                        reconstruction_type='l2'):
    """
    First compute the expected l_uvae data per data (line by line).
    Then take the average.
    Then take the inverse, as we want a loss.
    """
    assert reconstruction_type in ['riem', 'l2', 'l2_inner', 'bce']
    n_batch_data, data_dim = batch_data.shape
    assert batch_data.shape == batch_recon.shape, [
        batch_data.shape, batch_recon.shape]
    n_batch_data, dim = batch_data.shape

    if reconstruction_type == 'bce':
        bce_image = F.binary_cross_entropy(batch_data, batch_recon)
        bce_total = torch.sum(
             bce_image / batch_logvarx.exp() + 2 * batch_logvarx)
        bce_average = bce_total / n_batch_data
        return bce_average

    elif reconstruction_type == 'riem':
        n = int(np.sqrt(dim))
        batch_data_mat = batch_data.resize(n_batch_data, n, n)
        batch_recon_mat = batch_recon.resize(n_batch_data, n, n)

        assert is_spd(batch_data_mat)
        assert is_spd(batch_recon_mat)

        batch_logvarx = batch_logvarx.squeeze(dim=1)
        assert batch_logvarx.shape == (n_batch_data,)
        # Isotropic Gaussian
        scale_term = - data_dim / 2. * batch_logvarx
        sq_dist = riem_square_distance(batch_data, batch_recon)[:, 0]
        sq_dist_term = - sq_dist / (2. * batch_logvarx.exp())

    elif reconstruction_type == 'l2':
        if batch_logvarx.dim() > 1:
            batch_logvarx = batch_logvarx.squeeze(dim=1)
        if batch_logvarx.shape == (n_batch_data,):
            # Isotropic Gaussian
            scale_term = - data_dim / 2. * batch_logvarx
            sq_dist = torch.sum((batch_data - batch_recon) ** 2, dim=1)
            sq_dist_term = - sq_dist / (2. * batch_logvarx.exp())
        else:
            # Diagonal Gaussian
            assert batch_logvarx.shape == (
                n_batch_data, data_dim), batch_logvarx.shape
            scale_term = - 1. / 2. * torch.sum(batch_logvarx, dim=1)

            batch_varx = batch_logvarx.exp()
            norms = np.linalg.norm(batch_varx.cpu().detach().numpy(), axis=1)
            if np.isclose(norms, 0.).any():
                print('Warning: norms close to 0.')
            aux = (batch_data - batch_recon) ** 2 / batch_varx

            assert aux.shape == (n_batch_data, data_dim), aux.shape
            sq_dist_term = - 1. / 2. * torch.sum(aux, dim=1)

            for i in range(len(sq_dist_term)):
                if math.isinf(sq_dist_term[i]):
                    raise ValueError()

    # We keep the constant term to have an interpretation to the loss
    cst_term = - data_dim / 2. * torch.log(torch.Tensor([2 * np.pi]))
    cst_term = cst_term.to(DEVICE)
    assert scale_term.shape == (n_batch_data,)
    assert sq_dist_term.shape == (n_batch_data,), sq_dist_term
    l_uvae = cst_term + scale_term + sq_dist_term
    expected_l_uvae = torch.mean(l_uvae)

    # Make it a loss: -
    loss_reconstruction = -expected_l_uvae
    return loss_reconstruction


def regularization_loss(mu, logvar):
    n_batch_data, _ = logvar.shape
    assert logvar.shape == mu.shape
    loss_regularization = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    loss_regularization = torch.mean(loss_regularization)
    return loss_regularization


def neg_elbo(x, recon_x, logvarx, mu, logvar, reconstruction_type='l2'):
    recon_loss = reconstruction_loss(
        x, recon_x, logvarx, reconstruction_type)
    regu_loss = regularization_loss(mu, logvar)
    neg_elbo = recon_loss + regu_loss
    return neg_elbo


def neg_iwelbo_loss_base(
        x_expanded, recon_x_expanded,
        logvarx_expanded, mu_expanded, logvar_expanded, z_expanded,
        reconstruction_type='l2'):
    """
    The _expanded means that the tensor is of shape:
    n_is_samples x n_batch_data x tensor_dim.
    """
    assert not torch.isnan(x_expanded).any()
    assert not torch.isnan(recon_x_expanded).any()
    assert not torch.isnan(logvarx_expanded).any()
    assert not torch.isnan(mu_expanded).any()
    assert not torch.isnan(logvar_expanded).any()
    assert not torch.isnan(z_expanded).any()
    n_is_samples, n_batch_data, data_dim = x_expanded.shape
    _, _, latent_dim = mu_expanded.shape
    var_expanded = torch.exp(logvar_expanded)
    varx_expanded = torch.exp(logvarx_expanded)

    assert not torch.isnan(var_expanded).any()
    assert not torch.isnan(varx_expanded).any()
    log_QzGx = torch.sum(
        - 0.5 * (z_expanded - mu_expanded) ** 2 / var_expanded, dim=-1)
    assert not torch.isnan(log_QzGx).any()
    log_QzGx += torch.sum(- 0.5 * logvar_expanded, dim=-1)
    assert not torch.isnan(log_QzGx).any()
    log_QzGx += - 0.5 * latent_dim * torch.log(
        torch.Tensor([2 * np.pi])).to(DEVICE)
    assert not torch.isnan(log_QzGx).any()

    log_Pz = torch.sum(-0.5 * z_expanded ** 2, dim=-1)
    log_Pz += - 0.5 * latent_dim * torch.log(
        torch.Tensor([2 * np.pi])).to(DEVICE)[0]
    assert not torch.isnan(log_Pz).any()

    # These 4 lines are the reconstruction term: change here.
    if reconstruction_type == 'bce':
        #log_PxGz = torch.sum(
        #    x_expanded * torch.log(recon_x_expanded)
        #    + (1 - x_expanded) * torch.log(1 - recon_x_expanded),
        #    dim=-1)
        log_PxGz = -F.binary_cross_entropy(
            recon_x_expanded, x_expanded, reduction='none')
        log_PxGz = torch.sum(log_PxGz, dim=-1)
        assert log_PxGz.shape == (n_is_samples, n_batch_data), log_PxGz.shape
    else:
        log_PxGz = torch.sum(
            - 0.5 * (x_expanded - recon_x_expanded) ** 2 / varx_expanded
            - 0.5 * logvarx_expanded, dim=-1)
        log_PxGz += - 0.5 * data_dim * torch.log(
            torch.Tensor([2 * np.pi])).to(DEVICE)

    assert not torch.isnan(log_PxGz).any()
    log_weight = log_Pz + log_PxGz - log_QzGx
    assert log_weight.shape == (n_is_samples, n_batch_data)
    assert not torch.isnan(log_weight).any()

    iwelbo = log_mean_exp(log_weight, dim=0)
    assert not torch.isnan(iwelbo).any()
    assert iwelbo.shape == (n_batch_data,)

    iwelbo = torch.mean(iwelbo)
    assert not torch.isnan(iwelbo).any()
    neg_iwelbo = -iwelbo
    return neg_iwelbo


def neg_iwelbo(decoder, x, mu, logvar, n_is_samples,
               reconstruction_type='l2'):
    n_batch_data, latent_dim = mu.shape
    _, data_dim = x.shape

    mu_expanded = mu.expand(n_is_samples, n_batch_data, latent_dim)
    mu_expanded_flat = mu_expanded.resize(
        n_is_samples*n_batch_data, latent_dim)

    logvar_expanded = logvar.expand(n_is_samples, n_batch_data, -1)
    logvar_expanded_flat = logvar_expanded.resize(
        n_is_samples*n_batch_data, latent_dim)

    z_expanded_flat = toynn.sample_from_q(
        mu_expanded_flat, logvar_expanded_flat).to(DEVICE)
    z_expanded = z_expanded_flat.resize(
        n_is_samples, n_batch_data, latent_dim)

    batch_recon_expanded_flat, batch_logvarx_expanded_flat = decoder(
        z_expanded_flat)
    batch_recon_expanded = batch_recon_expanded_flat.resize(
        n_is_samples, n_batch_data, data_dim)

    batch_logvarx_expanded = batch_logvarx_expanded_flat.resize(
        n_is_samples, n_batch_data, batch_logvarx_expanded_flat.shape[-1])

    x_expanded = x.expand(
        n_is_samples, n_batch_data, data_dim)

    iwae = neg_iwelbo_loss_base(
        x_expanded,
        batch_recon_expanded, batch_logvarx_expanded,
        mu_expanded, logvar_expanded,
        z_expanded, reconstruction_type)
    return iwae
