

import torch
import torch.nn as nn
from torch.nn import functional as F

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


def reparametrize(mu, logvar, n_samples=1):
    n_batch_data, latent_dim = mu.shape

    std = logvar.mul(0.5).exp_()
    std_expanded = std.expand(
        n_samples, n_batch_data, latent_dim)
    mu_expanded = mu.expand(
        n_samples, n_batch_data, latent_dim)

    if CUDA:
        eps = torch.cuda.FloatTensor(
            n_samples, n_batch_data, latent_dim).normal_()
    else:
        eps = torch.FloatTensor(n_samples, n_batch_data, latent_dim).normal_()
    eps = torch.autograd.Variable(eps)

    z = eps * std_expanded + mu_expanded
    z_flat = z.resize(n_samples * n_batch_data, latent_dim)
    z_flat = z_flat.squeeze()  # case where latent_dim = 1: squeeze last dim
    return z_flat


def sample_from_q(mu, logvar, n_samples=1):
    return reparametrize(mu, logvar, n_samples)


def sample_from_prior(latent_dim, n_samples=1):
    if CUDA:
        mu = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
        logvar = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
    else:
        mu = torch.zeros(n_samples, latent_dim)
        logvar = torch.zeros(n_samples, latent_dim)
    return reparametrize(mu, logvar)


class Encoder(nn.Module):
    def __init__(self, latent_dim=2, data_dim=784):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.fc1 = nn.Linear(data_dim, 400)

        # Decrease amortization error
        self.fc1a = nn.Linear(400, 400)
        self.fc1b = nn.Linear(400, 400)
        self.fc1c = nn.Linear(400, 400)

        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.data_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1a(x))
        x = F.relu(self.fc1b(x))
        h1 = F.relu(self.fc1c(x))
        muz = self.fc21(h1)
        logvarz = self.fc22(h1)
        return muz, logvarz


class Decoder(nn.Module):
    def __init__(self, latent_dim=2, data_dim=784):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, data_dim)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h3))
        return recon_x, torch.zeros_like(recon_x)  # HACK


class VAE(nn.Module):
    """ Inspired by pytorch/examples VAE."""
    def __init__(self, latent_dim=20, data_dim=784):
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
        muz, logvarz = self.encoder(x)
        z = reparametrize(muz, logvarz)
        recon_x, _ = self.decoder(z)
        return recon_x, muz, logvarz


class EncoderCNN(nn.Module):
    def __init__(self, latent_dim=2, im_h=28, im_w=28):
        super(EncoderCNN, self).__init__()

        self.latent_dim = latent_dim
        self.im_h = im_h
        self.im_w = im_w

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=(4, 4), padding=(15, 15), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=(4, 4), padding=(15, 15), stride=2)
        self.fc11 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc12 = nn.Linear(in_features=1024, out_features=self.latent_dim)

        self.fc21 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc22 = nn.Linear(in_features=1024, out_features=self.latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        print('x=', x)
        x = x.view(-1, 1, 28, 28)
        print('x0=', x)
        x = self.relu(self.conv1(x))
        print('x1=', x)
        x = self.relu(self.conv2(x))
        print('x2=', x)
        x = x.view(-1, 128 * 28 * 28)
        print('x3=', x)

        muz = self.relu(self.fc11(x))
        print('muz1=', muz)
        muz = self.fc12(muz)
        print('muz2=', muz)

        logvarz = self.relu(self.fc21(x))
        print('logvarz1=', logvarz)
        logvarz = self.fc22(logvarz)
        print('logvarz2=', logvarz)

        return muz, logvarz


class DecoderCNN(nn.Module):
    def __init__(self, latent_dim=2, im_h=28, im_w=28):
        super(DecoderCNN, self).__init__()

        self.latent_dim = latent_dim
        self.im_h = im_h
        self.im_w = im_w

        self.fc1 = nn.Linear(in_features=self.latent_dim, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7 * 7 * 128)
        self.conv_t1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=4, padding=1, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=1,
            kernel_size=4, padding=1, stride=2)

    def forward(self, z):

        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.conv_t1(x))
        recon_x = F.sigmoid(self.conv_t2(x))
        recon_x = recon_x.view(-1, self.im_h * self.im_w)
        # Output flat recon_x
        return recon_x, torch.zeros_like(recon_x)  # HACK


class VAECNN(nn.Module):
    """
    Inspired by
    github.com/atinghosh/VAE-pytorch/blob/master/VAE_CNN_BCEloss.py.
    """
    def __init__(self, latent_dim=2, im_h=28, im_w=28):
        super(VAECNN, self).__init__()
        self.latent_dim = latent_dim
        self.im_h = im_h
        self.im_w = im_w

        self.encoder = EncoderCNN(
            latent_dim=latent_dim,
            im_h=im_h, im_w=im_w)

        self.decoder = DecoderCNN(
            latent_dim=latent_dim,
            im_h=im_h, im_w=im_w)

    def forward(self, x):
        muz, logvarz = self.encoder(x)
        z = reparametrize(muz, logvarz)
        recon_x, _ = self.decoder(z)
        return recon_x, muz, logvarz
