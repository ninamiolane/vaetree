"""NN fabric."""

import torch
import torch.autograd
import torch.nn as nn
import torch.optim
import torch.utils.data


CUDA = torch.cuda.is_available()


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()

    n_samples, latent_dim = mu.shape
    if CUDA:
        eps = torch.cuda.FloatTensor(n_samples, latent_dim).normal_()
    else:
        eps = torch.FloatTensor(n_samples, latent_dim).normal_()
    eps = torch.autograd.Variable(eps)
    z = eps * std + mu
    z = z.squeeze()
    return z


def sample_from_q(mu, logvar):
    return reparametrize(mu, logvar)


def sample_from_prior(latent_dim, n_samples=1):
    if CUDA:
        mu = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
        logvar = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
    else:
        mu = torch.zeros(n_samples, latent_dim)
        logvar = torch.zeros(n_samples, latent_dim)
    return reparametrize(mu, logvar)


class Encoder(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.fc1 = nn.Linear(
            in_features=data_dim, out_features=latent_dim)

        self.fc2 = nn.Linear(
            in_features=data_dim, out_features=latent_dim)

    def forward(self, x):
        """Forward pass of the encoder is encode."""
        mu = self.fc1(x)
        logvar = self.fc2(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, data_dim, n_layers=1, nonlinearity=False):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity

        # activation functions
        self.relu = nn.ReLU()

        # decoder
        self.d1 = nn.Linear(
            in_features=latent_dim, out_features=data_dim)
        self.dfc = nn.Linear(
            in_features=data_dim, out_features=data_dim)
        
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers-1):
            self.layers.append(self.dfc)


    def forward(self, z):
        """Forward pass of the decoder is to decode."""
        h = self.d1(z)
        if self.nonlinearity:
            h = self.relu(h)
        for i in range(self.n_layers-1):
            h = self.layers[i](h)
            if self.nonlinearity:
                h = self.relu(h)
        return h

    def generate(self, n_samples=1):
        """Generate from prior."""
        z = sample_from_prior(
            latent_dim=self.latent_dim, n_samples=n_samples)
        
        if n_samples == 1:
            z = z.unsqueeze(dim=0)
        else:
            z = z.unsqueeze(dim=1)
         
        x = self.forward(z)
        return z, x


class VAE(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.encoder = Encoder(
            latent_dim=latent_dim,
            data_dim=data_dim)

        self.decoder = Decoder(
            latent_dim=latent_dim,
            data_dim=data_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparametrize(mu, logvar)
        res = self.decoder(z)
        return res, mu, logvar