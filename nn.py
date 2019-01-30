"""NN fabric."""

import torch
import torch.utils.data
import torch.nn
import torch.optim
import torch.autograd


CUDA = torch.cuda.is_available()
W_IN = 160
H_IN = 192


def cnn_output_size(kernel_size, stride, padding, dilation, w_in, h_in):
    def one_dim(x):
        # From pytorch doc.
        return (((x + 2 * padding - dilation *
                  (kernel_size - 1) - 1) // stride) + 1)

    return one_dim(w_in), one_dim(h_in)


class VAE(torch.nn.Module):
    def __init__(self, n_channels, latent_dim):
        super(VAE, self).__init__()

        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # encoder
        self.e1 = torch.nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1)

        self.e1_width, self.e1_height = cnn_output_size(
            kernel_size=4,
            stride=2,
            padding=1,
            dilation=1,
            w_in=W_IN,
            h_in=H_IN)

        self.fc1 = torch.nn.Linear(
            in_features=64*self.e1_height*self.e1_width,
            out_features=latent_dim)

        self.fc2 = torch.nn.Linear(
            in_features=64*self.e1_height*self.e1_width,
            out_features=latent_dim)

        # decoder
        self.d1 = torch.nn.Linear(
            in_features=latent_dim,
            out_features=64*self.e1_width*self.e1_height)

        self.up1 = torch.nn.UpsamplingNearest2d(
            scale_factor=2)
        self.pd1 = torch.nn.ReplicationPad2d(1)
        self.d2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=self.n_channels,
            kernel_size=3,
            stride=1)

        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, x):
        a1 = self.e1(x)
        h1 = self.leakyrelu(a1)
        h1 = h1.view(-1, 64*self.e1_height*self.e1_width)

        return self.fc1(h1), self.fc2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 64, self.e1_width, self.e1_height)
        a1 = self.d2(self.pd1(self.up1(h1)))
        res = self.sigmoid(a1)
        return res

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


reconstruction_function = torch.nn.MSELoss()


def loss_function(recon_x, x, mu, logvar):
    mse = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(kld_element).mul_(-0.5)

    return mse + kld
