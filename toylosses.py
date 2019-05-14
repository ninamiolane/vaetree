"""Toy losses."""

import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


def fa_negloglikelihood(weight, data):
    weight = weight.cpu()
    sig2 = torch.mean(data ** 2, dim=0)

    loglikelihood_term_1 = - 1. / 2. * torch.log(
        2 * np.pi * (weight ** 2 + 1))
    loglikelihood_term_2 = - sig2 / (2 * (weight ** 2 + 1))
    loglikelihood = loglikelihood_term_1 + loglikelihood_term_2
    negloglikelihood = - loglikelihood
    return negloglikelihood


def reconstruction_loss(batch_data, batch_recon, batch_logvarx):
    """
    First compute the expected l_uvae data per data (line by line).
    Then take the average.
    Then take the inverse, as we want a loss.
    """
    n_batch_data, data_dim = batch_data.shape
    assert batch_data.shape == batch_recon.shape

    batch_logvarx = batch_logvarx.squeeze()
    if batch_logvarx.shape == (n_batch_data,):
        scale_term = - data_dim / 2. * batch_logvarx
        ssd = torch.sum((batch_data - batch_recon) ** 2, dim=1)
        ssd_term = - 1. / (2. * batch_logvarx.exp()) * ssd

    else:
        assert batch_logvarx.shape == (
            n_batch_data, data_dim), batch_logvarx.shape

        scale_term = - 1. / 2. * torch.sum(batch_logvarx, dim=1)

        batch_varx = batch_logvarx.exp()
        norms = np.linalg.norm(batch_varx.cpu().detach().numpy(), axis=1)
        if np.isclose(norms, 0.).any():
            print('Warning: norms close to 0.')
        aux = (batch_data - batch_recon) ** 2 / batch_varx

        if np.isinf(batch_data.cpu().detach().numpy()).any():
            raise ValueError('batch_data has a inf')
        if np.isinf(batch_recon.cpu().detach().numpy()).any():
            raise ValueError('batch_recon has a inf')
        if np.isinf(aux.cpu().detach().numpy()).any():
            raise ValueError('aux has a inf')
        assert aux.shape == (n_batch_data, data_dim), aux.shape
        ssd_term = - 1. / 2. * torch.sum(aux, dim=1)

        for i in range(len(ssd_term)):
            if math.isinf(ssd_term[i]):
                raise ValueError()

    # We keep the constant term to have an interpretation to the loss
    cst_term = - data_dim / 2. * torch.log(torch.Tensor([2 * np.pi]))
    cst_term = cst_term.to(DEVICE)
    assert scale_term.shape == (n_batch_data,)
    assert ssd_term.shape == (n_batch_data,), ssd_term
    l_uvae = cst_term + scale_term + ssd_term
    expected_l_uvae = torch.mean(l_uvae)
    loss_reconstruction = -expected_l_uvae
    return loss_reconstruction


def regularization_loss(mu, logvar):
    n_batch_data, _ = logvar.shape
    assert logvar.shape == mu.shape
    loss_regularization = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp())
    loss_regularization /= n_batch_data
    return loss_regularization


def iw_vae_loss(x, recon_x, logvarx, mu, logvar, z):
    var = torch.exp(logvar)
    varx = torch.exp(logvarx)

    log_QzGx = torch.sum(
        - 0.5 * (z - mu) ** 2 / var
        - 0.5 * logvar, dim=-1)
    log_QzGx += - 0.5 * torch.log(torch.Tensor([2 * np.pi]))

    log_Pz = torch.sum(-0.5 * z ** 2, dim=-1)
    log_Pz += - 0.5 * torch.log(torch.Tensor([2 * np.pi]))

    log_PxGz = torch.sum(
        - 0.5 * (x - recon_x) ** 2 / varx
        - 0.5 * logvarx, dim=-1)
    log_PxGz += - 0.5 * torch.log(torch.Tensor([2 * np.pi]))

    log_weight = log_Pz + log_PxGz - log_QzGx
    log_weight = log_weight - torch.max(log_weight, 0)[0]
    weight = torch.exp(log_weight)
    weight = weight / torch.sum(weight, dim=0)
    weight = Variable(weight.data, requires_grad=False)

    loss = -torch.mean(
        torch.sum(weight * (log_Pz + log_PxGz - log_QzGx), dim=0))
    return loss
