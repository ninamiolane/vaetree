"""NN fabric."""

import torch
import torch.autograd
import torch.nn as nn
import torch.optim
import torch.utils.data


CUDA = torch.cuda.is_available()

ENC_KS = 4
ENC_STR = 2
ENC_PAD = 1
ENC_DILATION = 1
ENC_C = 64

DEC_KS = 3
DEC_STR = 1
DEC_C = 64

DIS_KS = 4
DIS_STR = 2
DIS_PAD = 1
DIS_DILATION = 1
DIS_C = 64

# TODO(nina): Add Sequential for sequential layers
# TODO(nina): Use for loops to create layers in modules
# for a more compact code, use log2(image_size) for #layers.
# TODO(nina): Use nn.parallel to speed up?


def cnn_output_size(in_w, in_h, kernel_size=ENC_KS,
                    stride=ENC_STR,
                    padding=ENC_PAD,
                    dilation=ENC_DILATION):
    def one_dim(x):
        # From pytorch doc.
        return (((x + 2 * padding - dilation *
                  (kernel_size - 1) - 1) // stride) + 1)

    return one_dim(in_w), one_dim(in_h)


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
    def __init__(self, latent_dim, in_channels, in_h, in_w):
        super(Encoder, self).__init__()

        self.n_channels = in_channels
        self.latent_dim = latent_dim

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # encoder
        self.e1 = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=ENC_C,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn1 = nn.BatchNorm2d(self.e1.out_channels)

        self.w_e1, self.h_e1 = cnn_output_size(in_w=in_w, in_h=in_h)

        self.e2 = nn.Conv2d(
            in_channels=self.e1.out_channels,
            out_channels=ENC_C * 2,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn2 = nn.BatchNorm2d(self.e2.out_channels)
        self.w_e2, self.h_e2 = cnn_output_size(in_w=self.w_e1, in_h=self.h_e1)

        self.e3 = nn.Conv2d(
            in_channels=self.e2.out_channels,
            out_channels=ENC_C * 4,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn3 = nn.BatchNorm2d(self.e3.out_channels)
        self.w_e3, self.h_e3 = cnn_output_size(in_w=self.w_e2, in_h=self.h_e2)

        self.e4 = nn.Conv2d(
            in_channels=self.e3.out_channels,
            out_channels=ENC_C * 8,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn4 = nn.BatchNorm2d(self.e4.out_channels)

        self.w_e4, self.h_e4 = cnn_output_size(in_w=self.w_e3, in_h=self.h_e3)

        self.e5 = nn.Conv2d(
            in_channels=self.e4.out_channels,
            out_channels=ENC_C * 8,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn5 = nn.BatchNorm2d(self.e5.out_channels)
        self.w_e5, self.h_e5 = cnn_output_size(
            in_w=self.w_e4, in_h=self.h_e4)

        self.fcs_infeatures = self.e5.out_channels * self.h_e5 * self.w_e5
        self.fc1 = nn.Linear(
            in_features=self.fcs_infeatures, out_features=latent_dim)

        self.fc2 = nn.Linear(
            in_features=self.fcs_infeatures, out_features=latent_dim)

    def forward(self, x):
        """Forward pass of the encoder is encode."""
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.fcs_infeatures)
        mu = self.fc1(h5)
        logvar = self.fc2(h5)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, in_channels, in_h, in_w, out_channels):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_h = in_h
        self.in_w = in_w
        self.out_channels = out_channels

        self.fcs_infeatures = self.in_channels * self.in_h * self.in_w

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # decoder
        self.d1 = nn.Linear(
            in_features=latent_dim, out_features=self.fcs_infeatures)

        # TODO(johmathe): Get rid of warning.
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(
            in_channels=self.in_channels,
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
            out_channels=DEC_C * 2,
            kernel_size=DEC_KS,
            stride=DEC_STR)
        self.bnd4 = nn.BatchNorm2d(self.d5.out_channels, 1.e-3)

        # Generates recon
        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(
            in_channels=self.d5.out_channels,
            out_channels=self.out_channels,
            kernel_size=DEC_KS,
            stride=DEC_STR)

        # Generates scale_b
        self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd6 = nn.ReplicationPad2d(1)
        self.d7 = nn.Conv2d(
            in_channels=self.d5.out_channels,
            out_channels=self.out_channels,
            kernel_size=DEC_KS,
            stride=DEC_STR)

    def forward(self, z):
        """Forward pass of the decoder is to decode."""
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.in_channels, self.in_w, self.in_h)
        h2 = self.leakyrelu(self.bnd1(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bnd2(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bnd3(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bnd4(self.d5(self.pd4(self.up4(h4)))))
        h6 = self.d6(self.pd5(self.up5(h5)))
        h7 = self.d7(self.pd6(self.up6(h5)))
        recon = self.sigmoid(h6)
        scale_b = self.sigmoid(h7)
        return recon, scale_b


class VAE(nn.Module):
    def __init__(self, n_channels, latent_dim, in_w, in_h):
        super(VAE, self).__init__()

        self.n_channels = n_channels
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            latent_dim=self.latent_dim,
            in_channels=self.n_channels,
            in_h=in_h,
            in_w=in_w)

        dec_in_channels = self.encoder.e5.out_channels
        dec_in_h = self.encoder.h_e5
        dec_in_w = self.encoder.w_e5

        self.decoder = Decoder(
            latent_dim=latent_dim,
            in_channels=dec_in_channels,
            in_h=dec_in_h,
            in_w=dec_in_w,
            out_channels=self.n_channels)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparametrize(mu, logvar)
        res, scale_b = self.decoder(z)
        return res, scale_b, mu, logvar


class Discriminator(nn.Module):
    def __init__(self, latent_dim, in_channels, in_w, in_h):
        super(Discriminator, self).__init__()

        self.n_channels = in_channels
        self.latent_dim = latent_dim

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # discriminator
        self.dis1 = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=DIS_C,
            kernel_size=DIS_KS,
            stride=DIS_STR,
            padding=DIS_PAD)
        self.bn1 = nn.BatchNorm2d(self.dis1.out_channels)

        self.w_dis1, self.h_dis1 = cnn_output_size(in_w=in_w, in_h=in_h)

        self.dis2 = nn.Conv2d(
            in_channels=self.dis1.out_channels,
            out_channels=DIS_C * 2,
            kernel_size=DIS_KS,
            stride=DIS_STR,
            padding=DIS_PAD)
        self.bn2 = nn.BatchNorm2d(self.dis2.out_channels)
        self.w_dis2, self.h_dis2 = cnn_output_size(
            in_w=self.w_dis1, in_h=self.h_dis1)

        self.dis3 = nn.Conv2d(
            in_channels=self.dis2.out_channels,
            out_channels=DIS_C * 4,
            kernel_size=DIS_KS,
            stride=DIS_STR,
            padding=DIS_PAD)
        self.bn3 = nn.BatchNorm2d(self.dis3.out_channels)
        self.w_dis3, self.h_dis3 = cnn_output_size(
            in_w=self.w_dis2, in_h=self.h_dis2)

        self.dis4 = nn.Conv2d(
            in_channels=self.dis3.out_channels,
            out_channels=DIS_C * 8,
            kernel_size=DIS_KS,
            stride=DIS_STR,
            padding=DIS_PAD)
        self.bn4 = nn.BatchNorm2d(self.dis4.out_channels)
        self.w_dis4, self.h_dis4 = cnn_output_size(
            in_w=self.w_dis3, in_h=self.h_dis3)

        self.fcs_infeatures = (
            self.dis4.out_channels * self.h_dis4 * self.w_dis4)

        # Two layers to generate mu and log sigma2 of Gaussian
        # Distribution of features
        self.fc1 = nn.Linear(
            in_features=self.fcs_infeatures,
            out_features=self.fcs_infeatures)

        self.fc2 = nn.Linear(
            in_features=self.fcs_infeatures,
            out_features=self.fcs_infeatures)

        self.fc3 = nn.Linear(
            in_features=self.fcs_infeatures,
            out_features=1)

    def forward(self, x):
        """
        Forward pass of the discriminator is to take an image
        and output probability of the image being generated by the prior
        versus the learned approximation of the posterior.
        """
        h1 = self.leakyrelu(self.bn1(self.dis1(x)))
        h2 = self.leakyrelu(self.bn2(self.dis2(h1)))
        h3 = self.leakyrelu(self.bn3(self.dis3(h2)))
        h4 = self.leakyrelu(self.bn4(self.dis4(h3)))
        h4 = h4.view(-1, self.fcs_infeatures)
        h5_feature = self.fc1(h4)
        h5_logvar = self.fc2(h4)
        h6 = self.fc3(h5_feature)
        prob = self.sigmoid(h6)
        prob = prob.view(-1, 1)

        return prob, h5_feature, h5_logvar
