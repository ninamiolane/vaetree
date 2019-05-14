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
        result = toylosses.fa_negloglikelihood(weight, data)
        self.assertTrue(np.allclose(result, expected), result)

        weight = torch.Tensor([5.])
        data = torch.Tensor([1., 1.])
        expected = 2.56722
        result = toylosses.fa_negloglikelihood(weight, data)
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

if __name__ == '__main__':
    unittest.main()
