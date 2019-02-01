"""NN fabric."""

import torch
import torch.utils.data
import torch.nn
import torch.optim
import torch.autograd


CUDA = torch.cuda.is_available()

ENC_KERNEL_SIZE = 4
ENC_STRIDE = 2
ENC_PADDING = 1
ENC_DILATION = 1
ENC_CHANNEL_BASE = 64

DEC_KERNEL_SIZE = 3
DEC_STRIDE = 1
DEC_CHANNEL_BASE = 64


def cnn_output_size(w_in, h_in, kernel_size=ENC_KERNEL_SIZE,
                    stride=ENC_STRIDE,
                    padding=ENC_PADDING,
                    dilation=ENC_DILATION):
    def one_dim(x):
        # From pytorch doc.
        return (((x + 2 * padding - dilation *
                  (kernel_size - 1) - 1) // stride) + 1)

    return one_dim(w_in), one_dim(h_in)


class VAE(torch.nn.Module):
    # TODO(nina): Add BN in encoder and decoders?
    def __init__(self, n_channels, latent_dim, w_in, h_in):
        super(VAE, self).__init__()

        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # encoder
        self.e1 = torch.nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=ENC_CHANNEL_BASE,
            kernel_size=ENC_KERNEL_SIZE,
            stride=ENC_STRIDE,
            padding=ENC_PADDING)
        self.w_e1, self.h_e1 = cnn_output_size(
            w_in=w_in, h_in=h_in)

        self.e2 = torch.nn.Conv2d(
            in_channels=self.e1.out_channels,
            out_channels=ENC_CHANNEL_BASE*2,
            kernel_size=ENC_KERNEL_SIZE,
            stride=ENC_STRIDE,
            padding=ENC_PADDING)

        self.w_e2, self.h_e2 = cnn_output_size(
            w_in=self.w_e1, h_in=self.h_e1)

        self.elast = torch.nn.Conv2d(
            in_channels=self.e2.out_channels,
            out_channels=ENC_CHANNEL_BASE*4,
            kernel_size=ENC_KERNEL_SIZE,
            stride=ENC_STRIDE,
            padding=ENC_PADDING)
        self.w_elast, self.h_elast = cnn_output_size(
            w_in=self.w_e2, h_in=self.h_e2)

        self.fcs_infeatures = self.elast.out_channels*self.h_elast*self.w_elast
        self.fc1 = torch.nn.Linear(
            in_features=self.fcs_infeatures,
            out_features=latent_dim)

        self.fc2 = torch.nn.Linear(
            in_features=self.fcs_infeatures,
            out_features=latent_dim)

        # decoder
        self.d1 = torch.nn.Linear(
            in_features=latent_dim,
            out_features=self.fcs_infeatures)

        self.up1 = torch.nn.UpsamplingNearest2d(
            scale_factor=2)
        self.pd1 = torch.nn.ReplicationPad2d(1)
        self.d2 = torch.nn.Conv2d(
            in_channels=self.elast.out_channels,
            out_channels=DEC_CHANNEL_BASE*2,
            kernel_size=DEC_KERNEL_SIZE,
            stride=DEC_STRIDE)

        self.up2 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = torch.nn.ReplicationPad2d(1)
        self.d3 = torch.nn.Conv2d(
            in_channels=self.d2.out_channels,
            out_channels=DEC_CHANNEL_BASE,
            kernel_size=DEC_KERNEL_SIZE,
            stride=DEC_STRIDE)

        self.up3 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = torch.nn.ReplicationPad2d(1)
        self.d4 = torch.nn.Conv2d(
            in_channels=self.d3.out_channels,
            out_channels=self.n_channels,
            kernel_size=DEC_KERNEL_SIZE,
            stride=DEC_STRIDE)

        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.e1(x))
        h2 = self.leakyrelu(self.e2(h1))
        h3 = self.leakyrelu(self.elast(h2))
        h3 = h3.view(-1, self.fcs_infeatures)

        return self.fc1(h3), self.fc2(h3)

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
        h1 = h1.view(-1, self.elast.out_channels, self.w_elast, self.h_elast)
        h2 = self.leakyrelu(self.d2(self.pd1(self.up1(h1))))
        h3 = self.leakyrelu(self.d3(self.pd2(self.up2(h2))))
        h4 = self.d4(self.pd3(self.up3(h3)))
        return self.sigmoid(h4)

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
