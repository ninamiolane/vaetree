""" Unit tests for toylosses."""

import unittest
import numpy as np
import torch

import toylosses
import toynn

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

    def test_regularization_loss_1d(self):
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

    def test_regularization_loss_l_dim(self):
        mu = torch.Tensor([
            [1., 0.],
            [0., 0.],
            [1., 1.]])
        logvar = torch.Tensor([
            [1., 2.],
            [0., 0.],
            [1., 2.]])
        expected = 2.20245
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

    def test_neg_elbo_d_dim(self):
        n_batch_data = 3
        latent_dim = 1
        data_dim = 5

        x = torch.zeros((n_batch_data, data_dim)).to(DEVICE)
        recon_x = torch.zeros_like(x).to(DEVICE)
        logvarx = torch.zeros_like(x).to(DEVICE)

        mu = torch.zeros((n_batch_data, latent_dim)).to(DEVICE)
        logvar = torch.zeros_like(mu).to(DEVICE)

        expected = 0.5 * data_dim * np.log(2 * np.pi)
        result = toylosses.neg_elbo(x, recon_x, logvarx, mu, logvar)
        result = result.cpu().numpy()

        self.assertTrue(np.allclose(result, expected), result)

    def test_neg_elbo_d_dim_l_dim(self):
        n_batch_data = 3
        latent_dim = 4
        data_dim = 5

        x = torch.zeros((n_batch_data, data_dim)).to(DEVICE)
        recon_x = torch.zeros_like(x).to(DEVICE)
        logvarx = torch.zeros_like(x).to(DEVICE)

        mu = torch.zeros((n_batch_data, latent_dim)).to(DEVICE)
        logvar = torch.ones_like(mu).to(DEVICE)

        expected = (
            0.5 * data_dim * np.log(2 * np.pi)
            - 0.5 * latent_dim * (2 - np.exp(1)))
        result = toylosses.neg_elbo(x, recon_x, logvarx, mu, logvar)
        result = result.cpu().numpy()

        self.assertTrue(np.allclose(result, expected), result)

    def test_neg_iwelbo_d_dim(self):
        # Test all zeros - 1 iw sample
        n_is_samples = 1
        n_batch_data = 3
        latent_dim = 1
        data_dim = 5

        x_expanded = torch.zeros(
            (n_is_samples, n_batch_data, data_dim)).to(DEVICE)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros(
            (n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.zeros_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        expected = 0.5 * data_dim * np.log(2 * np.pi)
        result = toylosses.neg_iwelbo_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded)
        result = result.cpu().numpy()

        self.assertTrue(np.allclose(result, expected), result)

        # Test all zeros - multiple iw samples
        n_is_samples = 10
        n_batch_data = 3
        latent_dim = 1
        data_dim = 5

        x_expanded = torch.zeros(
            (n_is_samples, n_batch_data, data_dim)).to(DEVICE)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros(
            (n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.zeros_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        expected = 0.5 * data_dim * np.log(2 * np.pi)
        result = toylosses.neg_iwelbo_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded)
        result = result.cpu().numpy()

        self.assertTrue(np.allclose(result, expected), result)

    def test_neg_iwelbo_d_dim_l_dim(self):
        # Test all zeros, ones logvar- 1 iw sample
        n_is_samples = 1
        n_batch_data = 3
        latent_dim = 4
        data_dim = 5

        x_expanded = torch.zeros(
            (n_is_samples, n_batch_data, data_dim)).to(DEVICE)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros(
            (n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.ones_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        expected = 0.5 * data_dim * np.log(2 * np.pi) - 0.5 * latent_dim
        result = toylosses.neg_iwelbo_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded)
        result = result.cpu().numpy()

        self.assertTrue(np.allclose(result, expected), result)

        # Test all zeros, ones for logvar - multiple iw samples
        n_is_samples = 10
        n_batch_data = 3
        latent_dim = 4
        data_dim = 5

        x_expanded = torch.zeros(
            (n_is_samples, n_batch_data, data_dim)).to(DEVICE)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros(
            (n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.ones_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        expected = 0.5 * data_dim * np.log(2 * np.pi) - 0.5 * latent_dim
        result = toylosses.neg_iwelbo_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded)
        result = result.cpu().numpy()

        self.assertTrue(np.allclose(result, expected), result)

    def test_neg_iwelbo_d_dim_l_dim_bce(self):
        # Test all zeros, ones logvar- 1 iw sample
        n_is_samples = 1
        n_batch_data = 3
        latent_dim = 4
        data_dim = 5

        x_expanded = torch.zeros(
            (n_is_samples, n_batch_data, data_dim)).to(DEVICE)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros(
            (n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.ones_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        expected = - 2.
        result = toylosses.neg_iwelbo_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded,
            bce=True)
        result = result.cpu().numpy()

        self.assertTrue(np.allclose(result, expected), result)

        # Test all zeros, ones for logvar - multiple iw samples
        n_is_samples = 10
        n_batch_data = 3
        latent_dim = 4
        data_dim = 5

        x_expanded = torch.zeros(
            (n_is_samples, n_batch_data, data_dim)).to(DEVICE)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros(
            (n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.ones_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        expected = - 2.
        result = toylosses.neg_iwelbo_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded,
            bce=True)
        result = result.cpu().numpy()

        self.assertTrue(np.allclose(result, expected), result)

    def test_neg_iwelbo_loss_base(self):
        n_is_samples = 2
        n_batch_data = 3
        latent_dim = 1
        data_dim = 1

        # Test all zeros
        x_expanded = torch.zeros(
            (n_is_samples, n_batch_data, data_dim)).to(DEVICE)
        recon_x_expanded = torch.zeros_like(x_expanded).to(DEVICE)
        logvarx_expanded = torch.zeros_like(x_expanded).to(DEVICE)

        mu_expanded = torch.zeros(
            (n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.zeros_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        result = toylosses.neg_iwelbo_loss_base(
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

        mu_expanded = torch.zeros(
            (n_is_samples, n_batch_data, latent_dim)).to(DEVICE)
        logvar_expanded = torch.zeros_like(mu_expanded).to(DEVICE)
        z_expanded = torch.zeros_like(mu_expanded).to(DEVICE)

        result = toylosses.neg_iwelbo_loss_base(
            x_expanded, recon_x_expanded,
            logvarx_expanded, mu_expanded, logvar_expanded,
            z_expanded)
        result = result.cpu().numpy()

        expected = 1.80746
        self.assertTrue(np.allclose(result, expected), result)

    # def test_neg_iwelbo(self):
    #     # The expected result is on average the result for mu = 0.
    #     DATA_DIM = 1
    #     LATENT_DIM = 1
    #     N_DECODER_LAYERS = 1
    #     NONLINEARITY = False
    #     N_IS_SAMPLES = 3 #5000
    #     WITH_BIASX = False
    #     WITH_LOGVARX = False

    #     W_TRUE = {}
    #     B_TRUE = {}

    #     W_TRUE[0] = [[2.]]

    #     if WITH_LOGVARX:
    #         assert len(W_TRUE) == N_DECODER_LAYERS + 1, len(W_TRUE)
    #     else:
    #         assert len(W_TRUE) == N_DECODER_LAYERS

    #     decoder = toynn.make_decoder_true(
    #         w_true=W_TRUE, b_true=B_TRUE, latent_dim=LATENT_DIM,
    #         data_dim=DATA_DIM, n_layers=N_DECODER_LAYERS,
    #         nonlinearity=NONLINEARITY,
    #         with_biasx=WITH_BIASX, with_logvarx=WITH_LOGVARX)

    #     # Test all zeros
    #     # NB: Put z_expanded_flat all zeros in the toylosses' code
    #     #n_batch_data = 3
    #     #data_dim = DATA_DIM
    #     #latent_dim = LATENT_DIM

    #     #x = torch.zeros((n_batch_data, data_dim)).to(DEVICE)
    #     #mu = torch.zeros((n_batch_data, latent_dim)).to(DEVICE)
    #     #logvar = torch.zeros_like(mu).to(DEVICE)

    #     #expected = - (- 0.5 * np.log(2 * np.pi))
    #     #result = toylosses.iwae_loss(
    #     #    decoder, x, mu, logvar, n_is_samples=N_IS_SAMPLES)
    #     #result = result.detach().cpu().numpy()

    #     #self.assertTrue(np.allclose(result, expected), result)

    #     # Test with non-zero z
    #     # NB: Put:
    #     # z_expanded = torch.Tensor(
    #     #    [[[1.], [2.], [-1.]],
    #     #     [[0.], [-1.], [0.]]]
    #     #        ).to(DEVICE)
    #     # in the toylosses' code
    #     n_batch_data = 3
    #     data_dim = DATA_DIM
    #     latent_dim = LATENT_DIM
    #     N_IS_SAMPLES = 2

    #     x = torch.zeros((n_batch_data, data_dim)).to(DEVICE)
    #     mu = torch.zeros((n_batch_data, latent_dim)).to(DEVICE)
    #     logvar = torch.zeros_like(mu).to(DEVICE)

    #     expected = 1.7494846
    #     result = toylosses.iwae_loss(
    #         decoder, x, mu, logvar, n_is_samples=N_IS_SAMPLES)
    #     result = result.detach().cpu().numpy()

    #     print('expected = ', expected)
    #     print('result = ', result)
    #     self.assertTrue(np.allclose(result, expected), result)


if __name__ == '__main__':
    unittest.main()
