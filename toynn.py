"""NN fabric."""

import numpy as np

import torch
import torch.autograd
import torch.nn as nn
import torch.optim
import torch.utils.data


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


def make_decoder_true(w_true, b_true,
                      latent_dim, data_dim, n_layers,
                      nonlinearity=False, with_biasx=True, with_logvarx=True):
    decoder_true = Decoder(
        latent_dim=latent_dim, data_dim=data_dim,
        n_layers=n_layers,
        nonlinearity=nonlinearity,
        with_biasx=with_biasx,
        with_logvarx=with_logvarx)
    decoder_true.to(DEVICE)

    for i in range(n_layers):
        decoder_true.layers[i].weight.data = torch.tensor(
            w_true[i]).to(DEVICE)
        if with_biasx:
            decoder_true.layers[i].bias.data = torch.tensor(
                b_true[i]).to(DEVICE)

    if with_logvarx:
        # Layer predicting logvarx
        decoder_true.layers[n_layers].weight.data = torch.tensor(
            w_true[n_layers]).to(DEVICE)
        decoder_true.layers[n_layers].bias.data = torch.tensor(
            b_true[n_layers]).to(DEVICE)

    return decoder_true


def generate_from_decoder(decoder, n_samples):
    z, mux, logvarx = decoder.generate(n_samples=n_samples)
    _, data_dim = mux.shape

    mux = mux.cpu().detach().numpy()
    logvarx = logvarx.cpu().detach().numpy()

    generated_x = np.zeros((n_samples, data_dim))
    for i in range(n_samples):
        logvar = logvarx[i].squeeze()
        sigma = np.sqrt(np.exp((logvar)))
        eps = np.random.normal(
            loc=0, scale=sigma, size=(1, data_dim))
        generated_x[i] = mux[i] + eps

    return generated_x


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
    def __init__(self, latent_dim, data_dim,
                 with_biasz=True, with_logvarz=True):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.with_logvarz = with_logvarz

        self.fc1 = nn.Linear(
            in_features=data_dim, out_features=latent_dim, bias=with_biasz)

        if with_logvarz:
            self.fc2 = nn.Linear(
                in_features=data_dim, out_features=latent_dim)

    def forward(self, x):
        """Forward pass of the encoder is encode."""
        muz = self.fc1(x)
        if self.with_logvarz:
            logvarz = self.fc2(x)
        else:
            logvarz = torch.zeros_like(muz)

        return muz, logvarz


class Decoder(nn.Module):
    def __init__(self, latent_dim, data_dim, n_layers=1,
                 nonlinearity=False,
                 with_biasx=True, with_logvarx=True):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.with_logvarx = with_logvarx

        # activation functions
        self.relu = nn.ReLU()

        # layers
        self.layers = torch.nn.ModuleList()
        din = nn.Linear(
            in_features=latent_dim, out_features=data_dim, bias=with_biasx)
        self.layers.append(din)

        for i in range(self.n_layers-1):
            dfc = nn.Linear(
                in_features=data_dim, out_features=data_dim, bias=with_biasx)
            self.layers.append(dfc)

        # layer for logvarx
        if with_logvarx:
            if self.n_layers == 1:
                dlogvarx = nn.Linear(
                    in_features=latent_dim, out_features=data_dim)
            else:
                dlogvarx = nn.Linear(
                    in_features=data_dim, out_features=data_dim)
            self.layers.append(dlogvarx)

    def forward(self, z):
        """Forward pass of the decoder is to decode."""
        if self.latent_dim == 1 and len(z.shape) == 1:
            z = z.unsqueeze(-1)
        h = self.layers[0](z)
        if self.nonlinearity:
            h = self.relu(h)

        for i in range(1, self.n_layers-2):
            h = self.layers[i](h)
            if self.nonlinearity:
                h = self.relu(h)

        if self.n_layers == 1:
            x = self.layers[0](z)
            if self.with_logvarx:
                logvarx = self.layers[1](z)
            else:
                logvarx = torch.zeros_like(x)
        else:
            x = self.layers[self.n_layers-1](h)
            if self.with_logvarx:
                logvarx = self.layers[self.n_layers](h)
            else:
                logvarx = torch.zeros_like(x)
        return x, logvarx

    def generate(self, n_samples=1):
        """Generate from prior."""
        z = sample_from_prior(
            latent_dim=self.latent_dim, n_samples=n_samples)

        if n_samples == 1:
            z = z.unsqueeze(dim=0)
        else:
            z = z.unsqueeze(dim=1)

        x, logvarx = self.forward(z)
        return z, x, logvarx


class VAE(nn.Module):
    def __init__(self, latent_dim, data_dim, n_layers=1,
                 nonlinearity=False,
                 with_biasx=True, with_logvarx=True,
                 with_biasz=True, with_logvarz=True):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.encoder = Encoder(
            latent_dim=latent_dim,
            data_dim=data_dim,
            with_biasz=with_biasz,
            with_logvarz=with_logvarz)

        self.decoder = Decoder(
            latent_dim=latent_dim,
            data_dim=data_dim,
            n_layers=n_layers,
            nonlinearity=nonlinearity,
            with_biasx=with_biasx,
            with_logvarx=with_logvarx)

    def forward(self, x):
        muz, logvarz = self.encoder(x)
        z = reparametrize(muz, logvarz)
        res, logvarx = self.decoder(z)
        return res, logvarx, muz, logvarz


class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()

        self.data_dim = data_dim

        # activation functions
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        n_layers = int(np.log2(self.data_dim)) + 1  # HACK - at least 1 layers
        n_layers = 20

        self.layers = torch.nn.ModuleList()

        for i in range(n_layers):
            layer = nn.Linear(
                in_features=data_dim,
                out_features=data_dim)
            self.layers.append(layer)

        #for i in range(n_layers):
        #    layer = nn.Linear(
        #        in_features=int(data_dim / (2 ** i)),
        #        out_features=int(data_dim / (2 ** (i+1))))
        #    self.layers.append(layer)

        last_layer = nn.Linear(
            in_features=self.layers[-1].out_features,
            out_features=1)
        self.layers.append(last_layer)

    def forward(self, x):
        """
        Forward pass of the discriminator is to take an image
        and output probability of the image being generated by the prior
        versus the learned approximation of the posterior.
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        prob = self.sigmoid(h)

        return prob
