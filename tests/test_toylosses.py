""" Unit tests for toylosses."""

import unittest
import numpy as np
import torch

import toylosses

DEVICE = 'cuda'


class TestToylosses(unittest.TestCase):
    def test_fa_negloglikelihood(self):
        weight = torch.Tensor([2.])
        data = torch.Tensor([1., -1.])
        expected = 1.82366
        result = toylosses.fa_neg_loglikelihood(weight, data)
        self.assertTrue(np.allclose(result, expected), result)

        weight = torch.Tensor([5.])
        data = torch.Tensor([1., 1.])
        expected = 2.56722
        result = toylosses.fa_neg_loglikelihood(weight, data)
        self.assertTrue(np.allclose(result, expected), result)

    def test_regularization_loss(self):
        mu = torch.Tensor([[1.], [2.]])
        logvar = torch.Tensor([[0.], [0.]])
        expected = 1.25
        result = toylosses.regularization_loss(mu, logvar)
        self.assertTrue(np.allclose(result, expected), result)

        mu = torch.Tensor([[1.], [2.], [3.]])
        logvar = torch.Tensor([[0.], [0.], [0.]])
        expected = 7. / 3.
        result = toylosses.regularization_loss(mu, logvar)
        self.assertTrue(np.allclose(result, expected), result)

    def test_reconstruction_loss(self):
        batch_data = torch.tensor([[1.], [2.], [-1.]]).to(DEVICE)
        batch_recon = torch.tensor([[0.], [1.], [0.]]).to(DEVICE)

        batch_logvarx = torch.Tensor([[0.], [0.], [0.]]).to(DEVICE)
        expected = 1.41894
        result = toylosses.reconstruction_loss(
            batch_data, batch_recon, batch_logvarx)
        result = result.cpu().numpy()
        self.assertTrue(np.allclose(result, expected), result)

        batch_data = torch.tensor([[3.], [2.], [2.]]).to(DEVICE)
        batch_recon = torch.tensor([[0.], [0.], [0.]]).to(DEVICE)

        batch_logvarx = torch.Tensor([[0.], [0.], [0.]]).to(DEVICE)
        expected = 3.752272
        result = toylosses.reconstruction_loss(
            batch_data, batch_recon, batch_logvarx)
        result = result.cpu().numpy()
        self.assertTrue(np.allclose(result, expected), result)

    def test_iwae_base(self):
        n_is_samples = 2
        n_batch_data = 3
        latent_dim = 1
        data_dim = 1

        # Test all zeros
        x_expanded = torch.zeros((n_is_samples, n_batch_data, data_dim)).to(DEVICE)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros((n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.zeros_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        result = toylosses.iwae_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded)
        result = result.cpu().numpy()

        expected = - (- 0.5 * np.log(2 * np.pi))
        self.assertTrue(np.allclose(result, expected), result)

        # Test all zeros except x_expanded
        x_expanded = torch.Tensor(
            [[[1.], [2.], [3.]],
             [[2.], [-1.], [0.]]]
                ).to(DEVICE)
        assert x_expanded.shape == (n_is_samples, n_batch_data, data_dim)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros((n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.zeros_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        result = toylosses.iwae_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded)
        result = result.cpu().numpy()

        expected = 1.45118
        self.assertTrue(np.allclose(result, expected), result)



if __name__ == '__main__':
    unittest.main()
