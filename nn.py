"""NN fabric."""

import torch
import torch.utils.data
import torch.nn
import torch.optim
import torch.autograd


CUDA = torch.cuda.is_available()


class VAE(torch.nn.Module):
    def __init__(self, n_channels, ngf, ndf, latent_dim):
        super(VAE, self).__init__()

        self.n_channels = n_channels
        self.ngf = ngf
        self.ndf = ndf
        self.latent_dim = latent_dim

        # encoder
        self.e1 = torch.nn.Conv2d(n_channels, ndf, 4, 2, 1)
        self.bn1 = torch.nn.BatchNorm2d(ndf)

        self.fc1 = torch.nn.Linear(ndf*8*4*4, latent_dim)
        self.fc2 = torch.nn.Linear(ndf*8*4*4, latent_dim)

        # decoder
        self.d1 = torch.nn.Linear(latent_dim, ngf*8*2*4*4)

        self.up1 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = torch.nn.ReplicationPad2d(1)
        self.d2 = torch.nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn2 = torch.nn.BatchNorm2d(ngf*8, 1.e-3)

        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, x):
        a1 = self.e1(x)
        a2 = self.bn1(a1)
        h1 = self.leakyrelu(a2)
        h1 = h1.view(-1, self.ndf*8*4*4)

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
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn2(self.d2(self.pd1(self.up1(h1)))))

        return self.sigmoid(self.d2(self.pd1(self.up1(h2))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(
            x.view(-1, self.n_channels, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(
            x.view(-1, self.n_channels, self.ndf, self.ngf))
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
