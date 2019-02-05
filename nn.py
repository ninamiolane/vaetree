"""NN fabric."""

import torch
import torch.autograd
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.nn import functional as F


CUDA = torch.cuda.is_available()

ENC_KS = 4
ENC_STR = 2
ENC_PAD = 1
ENC_DILATION = 1
ENC_C = 64

DEC_KS = 3
DEC_STR = 1
DEC_C = 64


def cnn_output_size(w_in, h_in, kernel_size=ENC_KS,
                    stride=ENC_STR,
                    padding=ENC_PAD,
                    dilation=ENC_DILATION):
    def one_dim(x):
        # From pytorch doc.
        return (((x + 2 * padding - dilation *
                  (kernel_size - 1) - 1) // stride) + 1)

    return one_dim(w_in), one_dim(h_in)


class VAE(nn.Module):
    # TODO(nina): Add BN in encoder and decoders?
    def __init__(self, n_channels, latent_dim, w_in, h_in):
        super(VAE, self).__init__()

        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # encoder
        self.e1 = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=ENC_C,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn1 = nn.BatchNorm2d(self.e1.out_channels)

        self.w_e1, self.h_e1 = cnn_output_size(w_in=w_in, h_in=h_in)

        self.e2 = nn.Conv2d(
            in_channels=self.e1.out_channels,
            out_channels=ENC_C * 2,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn2 = nn.BatchNorm2d(self.e2.out_channels)
        self.w_e2, self.h_e2 = cnn_output_size(w_in=self.w_e1, h_in=self.h_e1)

        self.e3 = nn.Conv2d(
            in_channels=self.e2.out_channels,
            out_channels=ENC_C * 4,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn3 = nn.BatchNorm2d(self.e3.out_channels)
        self.w_e3, self.h_e3 = cnn_output_size(w_in=self.w_e2, h_in=self.h_e2)

        self.e4 = nn.Conv2d(
            in_channels=self.e3.out_channels,
            out_channels=ENC_C * 8,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn4 = nn.BatchNorm2d(self.e4.out_channels)
        self.w_e4, self.h_e4 = cnn_output_size(w_in=self.w_e3, h_in=self.h_e3)

        self.e5 = nn.Conv2d(
            in_channels=self.e4.out_channels,
            out_channels=ENC_C * 8,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn5 = nn.BatchNorm2d(self.e5.out_channels)
        self.w_e5, self.h_e5 = cnn_output_size(
            w_in=self.w_e4, h_in=self.h_e4)

        self.fcs_infeatures = self.e5.out_channels * self.h_e5 * self.w_e5
        self.fc1 = nn.Linear(
            in_features=self.fcs_infeatures, out_features=latent_dim)

        self.fc2 = nn.Linear(
            in_features=self.fcs_infeatures, out_features=latent_dim)

        # decoder
        self.d1 = nn.Linear(
            in_features=latent_dim, out_features=self.fcs_infeatures)

        # TODO(johmathe): Get rid of warning.
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(
            in_channels=self.e5.out_channels,
            out_channels=DEC_C * 16,
            kernel_size=DEC_KS,
            stride=DEC_STR)
        self.bnd1 = nn.BatchNorm2d(self.d2.out_channels, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(
            in_channels=self.d2.out_channels,
            out_channels=DEC_C * 8,
            kernel_size=DEC_KS,
            stride=DEC_STR)
        self.bnd2 = nn.BatchNorm2d(self.d3.out_channels, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(
            in_channels=self.d3.out_channels,
            out_channels=DEC_C * 4,
            kernel_size=DEC_KS,
            stride=DEC_STR)
        self.bnd3 = nn.BatchNorm2d(self.d4.out_channels, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(
            in_channels=self.d4.out_channels,
            out_channels= DEC_C * 2,
            kernel_size=DEC_KS,
            stride=DEC_STR)
        self.bnd4 = nn.BatchNorm2d(self.d5.out_channels, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(
            in_channels=self.d5.out_channels,
            out_channels=self.n_channels,
            kernel_size=DEC_KS,
            stride=DEC_STR)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.fcs_infeatures)

        return self.fc1(h5), self.fc2(h5)

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
        h1 = h1.view(-1, self.e5.out_channels, self.w_e5, self.h_e5)
        h2 = self.leakyrelu(self.bnd1(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bnd2(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bnd3(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bnd4(self.d5(self.pd4(self.up4(h4)))))
        h5 = self.d6(self.pd5(self.up3(h5)))
        return self.sigmoid(h5)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # Another hack alert...
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print('BCE: %s KLD: %s' % (bce.item(), kld.item()))
    return bce + kld
