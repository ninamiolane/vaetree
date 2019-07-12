"CNN modules."

import functools

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# Default NN parameters
DATA_DIM = 784
VAE_LATENT_DIM = 20

# Default CNN parameters
IM_C, IM_D, IM_H, IM_W = (1, 28, 28, 28)
CNN_DIM = 2
VAECNN_LATENT_DIM = 20

KS = (4, 4, 4)
STR = (4, 4, 4)
PAD = (6, 6, 6)
OUT_PAD = (0, 0, 0)  # No output padding
DIL = (2, 2, 2)  # No dilation
OUT_CHANNELS1 = 32
OUT_CHANNELS2 = 64
OUT_FC_FEATURES = 256
if CNN_DIM == 2:
    KS = KS[1:]
    STR = STR[1:]
    PAD = PAD[1:]
    DIL = DIL[1:]
    OUT_PAD = OUT_PAD[1:]


def conv_parameters(im_dim,
                    kernel_size=KS,
                    stride=STR,
                    padding=PAD,
                    dilation=DIL):

    if type(kernel_size) is int:
        kernel_size = np.repeat(kernel_size, im_dim)
    if type(stride) is int:
        stride = np.repeat(stride, im_dim)
    if type(padding) is int:
        padding = np.repeat(padding, im_dim)
    if type(dilation) is int:
        dilation = np.repeat(dilation, im_dim)

    assert len(kernel_size) == im_dim
    assert len(stride) == im_dim
    assert len(padding) == im_dim
    assert len(dilation) == im_dim

    return kernel_size, stride, padding, dilation


def conv_transpose_output_size(in_shape,
                               out_channels,
                               kernel_size=KS,
                               stride=STR,
                               padding=PAD,
                               output_padding=OUT_PAD,
                               dilation=DIL):
    im_dim = len(in_shape[1:])
    kernel_size, stride, padding, dilation = conv_parameters(
            im_dim, kernel_size, stride, padding, dilation)
    if type(output_padding) is int:
        output_padding = np.repeat(output_padding, im_dim)
    assert len(output_padding) == im_dim

    def one_dim(x):
        # From pytorch doc.
        output_shape_i_dim = (
            (in_shape[i_dim+1] - 1) * stride[i_dim]
            - 2 * padding[i_dim]
            + dilation[i_dim] * (kernel_size[i_dim] - 1)
            + output_padding[i_dim]
            + 1)
        return output_shape_i_dim

    out_shape = [one_dim(i_dim) for i_dim in range(im_dim)]
    out_shape = tuple(out_shape)

    return (out_channels,) + out_shape


def conv_transpose_input_size(out_shape,
                              in_channels,
                              kernel_size=KS,
                              stride=STR,
                              padding=PAD,
                              output_padding=OUT_PAD,
                              dilation=DIL):
    im_dim = len(out_shape[1:])
    kernel_size, stride, padding, dilation = conv_parameters(
            im_dim, kernel_size, stride, padding, dilation)
    if type(output_padding) is int:
        output_padding = np.repeat(output_padding, im_dim)

    def one_dim(i_dim):
        """Inverts the formula giving the output shape."""
        shape_i_dim = (
            ((out_shape[i_dim+1]
              + 2 * padding[i_dim]
              - dilation[i_dim] * (kernel_size[i_dim] - 1)
              - output_padding[i_dim] - 1)
             // stride[i_dim])
            + 1)

        assert shape_i_dim % 1 == 0, "CNN hyperparameters not valid."
        return int(shape_i_dim)

    in_shape = [one_dim(i_dim) for i_dim in range(im_dim)]
    in_shape = tuple(in_shape)

    return (in_channels,) + in_shape


def conv_output_size(in_shape,
                     out_channels,
                     kernel_size=KS,
                     stride=STR,
                     padding=PAD,
                     dilation=DIL):
    out_shape = conv_transpose_input_size(
        out_shape=in_shape,
        in_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=0,
        dilation=dilation)
    out_shape = (out_shape[0], out_shape[1], out_shape[2])
    return out_shape


def conv_input_size(out_shape,
                    in_channels,
                    kernel_size=KS,
                    stride=STR,
                    padding=PAD,
                    dilation=DIL):
    in_shape = conv_transpose_output_size(
        in_shape=out_shape,
        out_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=0,
        dilation=dilation)
    return in_shape


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
    z_flat = z_flat.squeeze(dim=1)  # case where latent_dim = 1: squeeze last dim
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
    def __init__(self, latent_dim=VAE_LATENT_DIM, data_dim=DATA_DIM):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.fc1 = nn.Linear(data_dim, latent_dim ** 2)

        # Decrease amortization error
        #self.fc1a = nn.Linear(400, 400)
        #self.fc1b = nn.Linear(400, 400)
        #self.fc1c = nn.Linear(400, 400)

        self.fc21 = nn.Linear(latent_dim ** 2, latent_dim)
        self.fc22 = nn.Linear(latent_dim ** 2, latent_dim)

        #self.leakyrelu = nn.LeakyReLU(0.2)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.data_dim)
        x = x.float()
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
        ##print('x = ', x)
        #n_batch_data, _ = x.shape
        #assert not torch.isnan(x).any()
        ##print('x1 = ', x)
        #assert not torch.isnan(x).any()
        #h1 = self.leakyrelu(self.fc1(x))
        ##print('x2 = ', x)
        #assert not torch.isnan(x).any()
        ##x = self.leakyrelu(self.fc1a(x))
        ##print('x3 = ', x)
        #assert not torch.isnan(x).any()
        ##x = self.leakyrelu(self.fc1b(x))
        ##print('x4 = ', x)
        #assert not torch.isnan(x).any()
        ##h1 = self.leakyrelu(self.fc1c(x))
        ##print('h1 = ', h1)
        #assert not torch.isnan(h1).any()
        #muz = self.fc21(h1)
        #assert not torch.isnan(muz).any()
        ##print('muz = ', muz)
        #logvarz = self.fc22(h1)
        #assert not torch.isnan(logvarz).any()
        ##print('logvarz = ', logvarz)
        #return muz, logvarz


class Decoder(nn.Module):
    def __init__(self, latent_dim=VAE_LATENT_DIM, data_dim=DATA_DIM):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        # TODO(nina): Find a better dim for intermediate activations
        self.fc3 = nn.Linear(
            in_features=latent_dim, out_features=latent_dim ** 2)
        self.fc4 = nn.Linear(
            in_features=latent_dim ** 2, out_features=data_dim)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h3))
        n_batch_data = recon_x.shape[0]
        return recon_x, torch.zeros(n_batch_data)  # HACK


class VAE(nn.Module):
    """ Inspired by pytorch/examples VAE."""
    def __init__(self, latent_dim=VAE_LATENT_DIM, data_dim=DATA_DIM):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.encoder = Encoder(
            latent_dim=latent_dim,
            data_dim=data_dim)

        self.decoder = Decoder(
            latent_dim=latent_dim,
            data_dim=data_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        muz, logvarz = self.encoder(x)
        z = self.reparametrize(muz, logvarz)
        recon_x, _ = self.decoder(z)
        return recon_x, muz, logvarz


def spd_layer(x):
    n_data = x.shape[0]
    n_channels = x.shape[1]
    sq_dist = torch.zeros(
        n_data, n_channels, n_channels)
    for i_channel in range(n_channels):
        for j_channel in range(i_channel):
            sq_dist[:, i_channel, j_channel] = torch.sum(
                (x[:, i_channel, :, :] - x[:, j_channel, :, :])**2)

    sigma2 = torch.mean(torch.sqrt(sq_dist)) ** 2
    sq_dist = sq_dist + sq_dist.permute(0, 2, 1)
    spd = torch.exp(- sq_dist / (2 * sigma2)).to(DEVICE)
    return spd


class EncoderCNN(nn.Module):
    def __init__(self, latent_dim=VAECNN_LATENT_DIM,
                 im_c=IM_C, im_d=IM_D, im_h=IM_H, im_w=IM_W,
                 cnn_dim=CNN_DIM):
        super(EncoderCNN, self).__init__()

        self.latent_dim = latent_dim
        self.im_c = im_c
        self.im_d = im_d
        self.im_h = im_h
        self.im_w = im_w
        self.cnn_dim = cnn_dim

        if CNN_DIM == 2:
            nn_conv = nn.Conv2d
            self.in_shape = (im_c, im_h, im_w)
        elif CNN_DIM == 3:
            nn_conv = nn.Conv3d
            self.in_shape = (im_c, im_d, im_h, im_w)
        else:
            raise ValueError('CNN_DIM is not 2D nor 3D.')

        self.conv1 = nn_conv(
            in_channels=self.im_c, out_channels=OUT_CHANNELS1,
            kernel_size=KS, padding=PAD, stride=STR)
        self.out_shape1 = conv_output_size(
            in_shape=self.in_shape,
            out_channels=self.conv1.out_channels,
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            dilation=self.conv1.dilation)

        self.conv2 = nn_conv(
            in_channels=self.conv1.out_channels, out_channels=OUT_CHANNELS2,
            kernel_size=KS, padding=PAD, stride=STR)
        self.out_shape2 = conv_output_size(
            in_shape=self.out_shape1,
            out_channels=self.conv2.out_channels,
            kernel_size=self.conv2.kernel_size,
            stride=self.conv2.stride,
            padding=self.conv2.padding,
            dilation=self.conv2.dilation)

        self.in_features = functools.reduce(
            (lambda x, y: x * y), self.out_shape2)
        self.fc11 = nn.Linear(
            in_features=self.in_features, out_features=OUT_FC_FEATURES)
        self.fc12 = nn.Linear(
            in_features=self.fc11.out_features, out_features=self.latent_dim)

        self.fc21 = nn.Linear(
            in_features=self.in_features, out_features=OUT_FC_FEATURES)
        self.fc22 = nn.Linear(
            in_features=self.fc21.out_features, out_features=self.latent_dim)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input flat x
        x = x.view((-1,) + self.in_shape)

        x = self.leakyrelu(self.conv1(x))
        assert x.shape[1:] == self.out_shape1
        x = self.leakyrelu(self.conv2(x))
        assert x.shape[1:] == self.out_shape2

        x = x.view(-1, self.in_features)

        muz = self.leakyrelu(self.fc11(x))
        muz = self.fc12(muz)

        logvarz = self.leakyrelu(self.fc21(x))
        logvarz = self.fc22(logvarz)

        return muz, logvarz


class DecoderCNN(nn.Module):
    def __init__(self, latent_dim=VAECNN_LATENT_DIM,
                 im_c=IM_C, im_d=IM_D, im_h=IM_H, im_w=IM_W,
                 cnn_dim=CNN_DIM, spd=False):

        super(DecoderCNN, self).__init__()

        self.latent_dim = latent_dim
        self.im_c = im_c
        self.im_d = im_d
        self.im_h = im_h
        self.im_w = im_w
        self.cnn_dim = cnn_dim
        self.spd = spd

        if CNN_DIM == 2:
            nn_conv_transpose = nn.ConvTranspose2d
            self.out_shape = (im_c, im_h, im_w)
        elif CNN_DIM == 3:
            nn_conv_transpose = nn.ConvTranspose3d
            self.out_shape = (im_c, im_d, im_h, im_w)
        else:
            raise ValueError('CNN_DIM is not 2D nor 3D.')

        # Layers given in reversed order

        # Conv transpose block (last)
        self.convt2_out_channels = self.im_c
        if spd:
            self.convt2_out_channels = self.im_w
        self.convt2 = nn_conv_transpose(
            in_channels=OUT_CHANNELS1, out_channels=self.convt2_out_channels,
            kernel_size=KS, padding=PAD, stride=STR)
        self.in_shape2 = conv_transpose_input_size(
            out_shape=self.out_shape,
            in_channels=self.convt2.in_channels,
            kernel_size=self.convt2.kernel_size,
            stride=self.convt2.stride,
            padding=self.convt2.padding,
            dilation=self.convt2.dilation)

        self.convt1 = nn_conv_transpose(
            in_channels=OUT_CHANNELS2, out_channels=OUT_CHANNELS1,
            kernel_size=KS, padding=PAD, stride=STR)
        self.in_shape1 = conv_transpose_input_size(
            out_shape=self.in_shape2,
            in_channels=self.convt1.in_channels,
            kernel_size=self.convt1.kernel_size,
            stride=self.convt1.stride,
            padding=self.convt1.padding,
            dilation=self.convt1.dilation)

        # Fully connected block (first)
        self.out_features = functools.reduce(
            (lambda x, y: x * y), self.in_shape1)
        self.fc2 = nn.Linear(
            in_features=OUT_FC_FEATURES, out_features=self.out_features)

        self.fc1 = nn.Linear(
            in_features=self.latent_dim, out_features=self.fc2.in_features)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        assert z.shape[1:] == (self.latent_dim,)
        x = F.elu(self.fc1(z))
        assert x.shape[1:] == (self.fc2.in_features,)
        x = F.elu(self.fc2(x))
        assert x.shape[1:] == (self.out_features,)

        x = x.view((-1,) + self.in_shape1)
        assert x.shape[1:] == self.in_shape1
        x = self.leakyrelu(self.convt1(x, output_size=self.in_shape2[1:]))

        assert x.shape[1:] == self.in_shape2
        x = self.sigmoid(self.convt2(x, output_size=self.out_shape[1:]))

        if self.spd:
            x = spd_layer(x)

        # Output flat recon_x
        # Note: this also multiplies the channels, assuming that im_c=1.
        # TODO(nina): Bring back the channels as their full dimension
        out_dim = functools.reduce((lambda x, y: x * y), self.out_shape)
        recon_x = x.view(-1, out_dim)
        return recon_x, torch.zeros_like(recon_x)  # HACK


class VAECNN(nn.Module):
    """
    Inspired by
    github.com/atinghosh/VAE-pytorch/blob/master/VAE_CNN_BCEloss.py.
    """
    def __init__(self, latent_dim=VAECNN_LATENT_DIM,
                 im_c=IM_C, im_d=IM_D, im_h=IM_H, im_w=IM_W,
                 cnn_dim=CNN_DIM, spd=False):

        super(VAECNN, self).__init__()

        self.latent_dim = latent_dim
        self.im_c = im_c
        self.im_d = im_d
        self.im_h = im_h
        self.im_w = im_w
        self.cnn_dim = cnn_dim
        self.spd = spd

        self.encoder = EncoderCNN(
            latent_dim=self.latent_dim,
            im_c=self.im_c,
            im_d=self.im_d,
            im_h=self.im_h,
            im_w=self.im_w,
            cnn_dim=self.cnn_dim)

        self.decoder = DecoderCNN(
            latent_dim=self.latent_dim,
            im_c=self.im_c,
            im_d=self.im_d,
            im_h=self.im_h,
            im_w=self.im_w,
            cnn_dim=self.cnn_dim,
            spd=self.spd)

    def forward(self, x):
        muz, logvarz = self.encoder(x)
        z = reparametrize(muz, logvarz)
        recon_x, _ = self.decoder(z)
        return recon_x, muz, logvarz
