""" Unit tests for performance metrics."""

import unittest
import numpy as np

import metrics


class TestMetrics(unittest.TestCase):
    def test_mutual_information(self):
        # Two constant images
        data = 3 * np.ones((2, 2))
        recon = 4 * np.ones((2, 2))
        expected = 0.
        result = metrics.mutual_information(recon, data)

        self.assertTrue(np.allclose(result, expected))

    def test_frechet_inception_distance(self):
        n_imgs = 5
        # Two constant images:
        # Covariances are 0
        # FID = mse between the means
        data = 3 * np.ones((2, 2))
        recon = 4 * np.ones((2, 2))

        data_set = np.repeat(data, repeats=n_imgs, axis=0)
        recon_set = np.repeat(recon, repeats=n_imgs, axis=0)
        data_mean = data.ravel()
        recon_mean = recon.ravel()

        expected = (data_mean-recon_mean)**2
        result = metrics.frechet_inception_distance(recon_set, data_set)

        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
    unittest.main()
