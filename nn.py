"""
NN fabric.

Using Pytorch convention: (N, C, D, H, W).
Image dim refers to len((C, D, H, W)): with channels
Conv dim refers to len((D, H, W)): no_channels
Data dim is the length of the vector of the flatten data:
    C x D x H x W
"""

import functools
import numpy as np
import torch
import torch.autograd
from torch.nn import functional as F
import torch.nn as nn
import torch.optim
import torch.utils.data


CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')


# For VaeConv modules

KS = 4
STR = 4
PAD = 6
OUT_PAD = 0
DIL = 2
OUT_CHANNELS1 = 32
OUT_CHANNELS2 = 64
OUT_FC_FEATURES = 256


# For VaeConvPlus modules

ENC_KS = 4
ENC_STR = 2
ENC_PAD = 1
ENC_DIL = 1
ENC_C = 64

DEC_KS = 3
DEC_STR = 1
DEC_PAD = 0
DEC_DIL = 1
DEC_C = 64

DIS_KS = 4
DIS_STR = 2
DIS_PAD = 1
DIS_DIL = 1
DIS_C = 64


# Conv Layers

NN_CONV = {
    2: nn.Conv2d,
    3: nn.Conv3d}
NN_CONV_TRANSPOSE = {
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d}

# TODO(nina): Add Sequential for sequential layers
# TODO(nina): Use for loops to create layers in modules
# for a more compact code, use log2(image_size) for #layers.
# TODO(nina): Use nn.parallel to speed up?
# TODO(nina): Uniformize flattened/unflattened inputs/outputs


def conv_parameters(conv_dim,
                    kernel_size=KS,
                    stride=STR,
                    padding=PAD,
                    dilation=DIL):

    if type(kernel_size) is int:
        kernel_size = np.repeat(kernel_size, conv_dim)
    if type(stride) is int:
        stride = np.repeat(stride, conv_dim)
    if type(padding) is int:
        padding = np.repeat(padding, conv_dim)
    if type(dilation) is int:
        dilation = np.repeat(dilation, conv_dim)

    assert len(kernel_size) == conv_dim
    assert len(stride) == conv_dim
    assert len(padding) == conv_dim
    assert len(dilation) == conv_dim

    return kernel_size, stride, padding, dilation


def conv_transpose_output_size(in_shape,
                               out_channels,
                               kernel_size=KS,
                               stride=STR,
                               padding=PAD,
                               output_padding=OUT_PAD,
                               dilation=DIL):
    conv_dim = len(in_shape[1:])
    kernel_size, stride, padding, dilation = conv_parameters(
            conv_dim, kernel_size, stride, padding, dilation)
    if type(output_padding) is int:
        output_padding = np.repeat(output_padding, conv_dim)
    assert len(output_padding) == conv_dim

    def one_dim(x):
        # From pytorch doc.
        output_shape_i_dim = (
            (in_shape[i_dim+1] - 1) * stride[i_dim]
            - 2 * padding[i_dim]
            + dilation[i_dim] * (kernel_size[i_dim] - 1)
            + output_padding[i_dim]
            + 1)
        return output_shape_i_dim

    out_shape = [one_dim(i_dim) for i_dim in range(conv_dim)]
    out_shape = tuple(out_shape)

    return (out_channels,) + out_shape


def conv_transpose_input_size(out_shape,
                              in_channels,
                              kernel_size=KS,
                              stride=STR,
                              padding=PAD,
                              output_padding=OUT_PAD,
                              dilation=DIL):
    conv_dim = len(out_shape[1:])
    kernel_size, stride, padding, dilation = conv_parameters(
            conv_dim, kernel_size, stride, padding, dilation)
    if type(output_padding) is int:
        output_padding = np.repeat(output_padding, conv_dim)

    def one_dim(i_dim):
        """Inverts the formula giving the output shape."""
        shape_i_dim = (
            ((out_shape[i_dim+1]
              + 2 * padding[i_dim]
              - dilation[i_dim] * (kernel_size[i_dim] - 1)
              - output_padding[i_dim] - 1)
             // stride[i_dim])
            + 1)

        assert shape_i_dim % 1 == 0, "Conv hyperparameters not valid."
        return int(shape_i_dim)

    in_shape = [one_dim(i_dim) for i_dim in range(conv_dim)]
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
    # Case where latent_dim = 1: squeeze last dim
    z_flat = z_flat.squeeze(dim=1)
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


def kernel_aggregation(x):
    """
    From: https://arxiv.org/pdf/1711.06540.pdf.
    """
    n_data, n_channels, _, _ = x.shape

    sq_dist = torch.zeros(n_data, n_channels, n_channels)
    for i_channel in range(n_channels):
        for j_channel in range(i_channel):
            sq_dist_aux = torch.sum(
                (x[:, i_channel, :, :] - x[:, j_channel, :, :])**2, dim=-1)
            sq_dist_aux = torch.sum(sq_dist_aux, dim=-1)
            assert sq_dist_aux.shape == (n_data,), sq_dist_aux.shape
            sq_dist[:, i_channel, j_channel] = sq_dist_aux

    sigma2 = torch.mean(sq_dist)
    sq_dist = 2 * (sq_dist + sq_dist.permute(0, 2, 1))  # -- HACK
    spd = torch.exp(- sq_dist / (2 * sigma2)).to(DEVICE)
    return spd


class Encoder(nn.Module):
    def __init__(self, latent_dim, data_dim, inner_dim=4096):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.fc10 = nn.Linear(data_dim, inner_dim)
        self.fc11 = nn.Linear(inner_dim, inner_dim)
        self.fc12 = nn.Linear(inner_dim, inner_dim)

        # Decrease amortization error with fc1a, fc1b, etc if needed.

        self.fc21 = nn.Linear(inner_dim, latent_dim)
        self.fc22 = nn.Linear(inner_dim, latent_dim)

    def forward(self, x):
        # Note: no channels
        """Encoder flattens the data, as its first step."""
        x = x.view(-1, self.data_dim).float()
        h1 = F.relu(self.fc10(x))
        h1 = F.relu(self.fc11(h1))
        h1 = F.relu(self.fc12(h1))
        return self.fc21(h1), self.fc22(h1)


class Decoder(nn.Module):
    def __init__(self, latent_dim, data_dim, with_sigmoid,
                 n_layers=4, inner_dim=128, with_skip=False, logvar=0.):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.with_sigmoid = with_sigmoid
        self.n_layers = n_layers
        self.inner_dim = inner_dim
        self.with_skip = with_skip
        self.logvar = logvar
        add_dim = 0
        if with_skip:
            add_dim = latent_dim

        self.layers = torch.nn.ModuleList()

        fc_in = nn.Linear(
            in_features=latent_dim, out_features=inner_dim)
        self.layers.append(fc_in)

        for i in range(self.n_layers - 2):
            fc = nn.Linear(
                in_features=inner_dim+add_dim, out_features=inner_dim)
            self.layers.append(fc)

        fc_out = nn.Linear(
            in_features=inner_dim+add_dim, out_features=data_dim)
        self.layers.append(fc_out)

    def forward(self, z):
        h = z

        for i in range(self.n_layers - 1):
            h = torch.sigmoid(self.layers[i](h))
            if self.with_skip:
                h = torch.cat([h, z], dim=1)

        recon_x = self.layers[self.n_layers-1](h)
        if self.with_sigmoid:
            recon_x = torch.sigmoid(recon_x)

        n_batch_data = recon_x.shape[0]
        logvar_x = self.logvar * torch.ones((n_batch_data, 1)).to(DEVICE)
        return recon_x, logvar_x


class Vae(nn.Module):
    """ Inspired by pytorch/examples Vae."""
    def __init__(self, latent_dim, data_dim, with_sigmoid,
                 n_layers=4, inner_dim=128, with_skip=False, logvar=0.):
        super(Vae, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.with_sigmoid = with_sigmoid
        self.n_layers = n_layers
        self.inner_dim = inner_dim
        self.with_skip = with_skip
        self.logvar = logvar

        self.encoder = Encoder(
            latent_dim=latent_dim,
            data_dim=data_dim,
            inner_dim=inner_dim)

        self.decoder = Decoder(
            latent_dim=latent_dim,
            data_dim=data_dim,
            with_sigmoid=with_sigmoid,
            n_layers=n_layers,
            inner_dim=inner_dim,
            with_skip=with_skip,
            logvar=logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        muz, logvarz = self.encoder(x)
        z = self.reparametrize(muz, logvarz)
        recon_x, _ = self.decoder(z)
        return recon_x, muz, logvarz


class EncoderConv(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(EncoderConv, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.conv_dim = len(img_shape[1:])
        self.nn_conv = NN_CONV[self.conv_dim]

        self.conv1 = self.nn_conv(
            in_channels=self.img_shape[0], out_channels=OUT_CHANNELS1,
            kernel_size=KS, padding=PAD, stride=STR)
        self.out_shape1 = conv_output_size(
            in_shape=self.img_shape,
            out_channels=self.conv1.out_channels,
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            dilation=self.conv1.dilation)

        self.conv2 = self.nn_conv(
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
        """Unflattens to put in img shape"""
        x = x.view((-1,) + self.img_shape)

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


class DecoderConv(nn.Module):
    def __init__(self, latent_dim, img_shape, with_sigmoid,
                 riem_log_exp=False):

        super(DecoderConv, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.with_sigmoid = with_sigmoid

        self.conv_dim = len(img_shape[1:])
        self.riem_log_exp = riem_log_exp
        self.nn_conv_transpose = NN_CONV_TRANSPOSE[self.conv_dim]

        # Layers given in reversed order

        # Conv transpose block (last)
        self.convt2_out_channels = self.img_shape[0]
        if self.riem_log_exp:
            self.convt2_out_channels = self.img_shape[1]
        self.convt2 = self.nn_conv_transpose(
            in_channels=OUT_CHANNELS1, out_channels=self.convt2_out_channels,
            kernel_size=KS, padding=PAD, stride=STR)
        self.in_shape2 = conv_transpose_input_size(
            out_shape=self.img_shape,
            in_channels=self.convt2.in_channels,
            kernel_size=self.convt2.kernel_size,
            stride=self.convt2.stride,
            padding=self.convt2.padding,
            dilation=self.convt2.dilation)

        self.convt1 = self.nn_conv_transpose(
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
        """
        Outputs flattened image.
        As the image is seen flattened when considering the variance.
        """
        assert z.shape[1:] == (self.latent_dim,)
        x = F.elu(self.fc1(z))
        assert x.shape[1:] == (self.fc2.in_features,)
        x = F.elu(self.fc2(x))
        assert x.shape[1:] == (self.out_features,)

        x = x.view((-1,) + self.in_shape1)
        assert x.shape[1:] == self.in_shape1
        x = self.leakyrelu(self.convt1(x, output_size=self.in_shape2[1:]))

        assert x.shape[1:] == self.in_shape2
        if self.riem_log_exp:
            x = self.convt2(x, output_size=self.img_shape[1:])
            x = kernel_aggregation(x)
        else:
            x = self.convt2(x, output_size=self.img_shape[1:])
            if self.with_sigmoid:
                x = self.sigmoid(x)

        # Output flat recon_x
        # Note: this also multiplies the channels, assuming that img_c=1.
        # TODO(nina): Bring back the channels as their full dimension

        out_dim = functools.reduce((lambda x, y: x * y), self.img_shape)
        recon_x = x.view(-1, out_dim)

        return recon_x, torch.zeros((recon_x.shape[0], 1)).to(DEVICE)  # HACK


class VaeConv(nn.Module):
    """
    Inspired by
    github.com/atinghosh/Vae-pytorch/blob/master/Vae_Conv_BCEloss.py.
    """
    def __init__(self, latent_dim, img_shape, with_sigmoid,
                 riem_log_exp=False):

        super(VaeConv, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.with_sigmoid = with_sigmoid
        self.riem_log_exp = riem_log_exp

        self.encoder = EncoderConv(
            latent_dim=self.latent_dim,
            img_shape=self.img_shape)

        self.decoder = DecoderConv(
            latent_dim=self.latent_dim,
            img_shape=self.img_shape,
            with_sigmoid=with_sigmoid,
            riem_log_exp=self.riem_log_exp)

    def forward(self, x):
        muz, logvarz = self.encoder(x)
        z = reparametrize(muz, logvarz)
        recon_x, _ = self.decoder(z)
        return recon_x, muz, logvarz


class EncoderConvPlus(nn.Module):

    def enc_conv_output_size(self, in_shape, out_channels):
        return conv_output_size(
                in_shape, out_channels,
                kernel_size=ENC_KS,
                stride=ENC_STR,
                padding=ENC_PAD,
                dilation=ENC_DIL)

    def __init__(self, latent_dim, img_shape):
        super(EncoderConvPlus, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=self.img_shape[0],
            out_channels=ENC_C,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn1 = nn.BatchNorm2d(self.enc1.out_channels)

        self.enc1_out_shape = self.enc_conv_output_size(
            in_shape=self.img_shape,
            out_channels=self.enc1.out_channels)

        self.enc2 = nn.Conv2d(
            in_channels=self.enc1.out_channels,
            out_channels=ENC_C * 2,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn2 = nn.BatchNorm2d(self.enc2.out_channels)

        self.enc2_out_shape = self.enc_conv_output_size(
            in_shape=self.enc1_out_shape,
            out_channels=self.enc2.out_channels)

        self.enc3 = nn.Conv2d(
            in_channels=self.enc2.out_channels,
            out_channels=ENC_C * 4,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn3 = nn.BatchNorm2d(self.enc3.out_channels)

        self.enc3_out_shape = self.enc_conv_output_size(
            in_shape=self.enc2_out_shape,
            out_channels=self.enc3.out_channels)

        self.enc4 = nn.Conv2d(
            in_channels=self.enc3.out_channels,
            out_channels=ENC_C * 8,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn4 = nn.BatchNorm2d(self.enc4.out_channels)

        self.enc4_out_shape = self.enc_conv_output_size(
            in_shape=self.enc3_out_shape,
            out_channels=self.enc4.out_channels)

        self.enc5 = nn.Conv2d(
            in_channels=self.enc4.out_channels,
            out_channels=ENC_C * 8,
            kernel_size=ENC_KS,
            stride=ENC_STR,
            padding=ENC_PAD)
        self.bn5 = nn.BatchNorm2d(self.enc5.out_channels)

        self.enc5_out_shape = self.enc_conv_output_size(
            in_shape=self.enc4_out_shape,
            out_channels=self.enc5.out_channels)

        self.fcs_infeatures = functools.reduce(
            (lambda x, y: x * y), self.enc5_out_shape)
        self.fc1 = nn.Linear(
            in_features=self.fcs_infeatures, out_features=latent_dim)

        self.fc2 = nn.Linear(
            in_features=self.fcs_infeatures, out_features=latent_dim)

    def forward(self, x):
        """Forward pass of the encoder is encode."""
        h1 = self.leakyrelu(self.bn1(self.enc1(x)))
        h2 = self.leakyrelu(self.bn2(self.enc2(h1)))
        h3 = self.leakyrelu(self.bn3(self.enc3(h2)))
        h4 = self.leakyrelu(self.bn4(self.enc4(h3)))
        h5 = self.leakyrelu(self.bn5(self.enc5(h4)))
        h5 = h5.view(-1, self.fcs_infeatures)
        mu = self.fc1(h5)
        logvar = self.fc2(h5)

        return mu, logvar


class DecoderConvPlus(nn.Module):
    def conv_output_size(self, in_shape, out_channels):
        return conv_output_size(
                in_shape, out_channels,
                kernel_size=DEC_KS,
                stride=DEC_STR,
                padding=DEC_PAD,
                dilation=DEC_DIL)

    def block(self, block_id, in_shape,
              channels_fact, scale_factor=2, pad=1):
        conv_dim = len(in_shape[1:])
        nn_conv = NN_CONV[conv_dim]
        in_channels = in_shape[0]

        up = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        pd = nn.ReplicationPad2d(pad)
        conv = nn_conv(
            in_channels=in_channels,
            out_channels=DEC_C * channels_fact,
            kernel_size=DEC_KS,
            stride=DEC_STR)
        bn = nn.BatchNorm2d(conv.out_channels, 1.e-3)

        out_shape = self.conv_output_size(
            in_shape=(in_channels,
                      scale_factor*self.in_shape[1] + 2*pad,
                      scale_factor*self.in_shape[2] + 2*pad),
            out_channels=conv.out_channels)
        return up, pd, conv, bn, out_shape

    def end_block(self, block_id, in_shape, scale_factor=2, pad=1):
        up = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        pd = nn.ReplicationPad2d(pad)
        conv = nn.Conv2d(
            in_channels=in_shape[0],
            out_channels=self.img_shape[0],
            kernel_size=DEC_KS,
            stride=DEC_STR)
        # TODO(nina): put last resampling at other position?
        end_up = nn.UpsamplingNearest2d(size=self.img_shape[1:])
        return up, pd, conv, end_up

    def __init__(self, latent_dim, img_shape, with_sigmoid, in_shape=None):
        super(DecoderConvPlus, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.with_sigmoid = with_sigmoid
        # TODO(nina): Remove in_shape by propagating img dimensions
        self.in_shape = in_shape

        self.fcs_infeatures = functools.reduce(
            (lambda x, y: x * y), self.in_shape)

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # decoder
        self.l0 = nn.Linear(
            in_features=latent_dim, out_features=self.fcs_infeatures)

        self.up1, self.pd1, self.conv1, self.bn1, self.out_shape1 = self.block(
            block_id=1, in_shape=self.in_shape, channels_fact=8)

        self.up2, self.pd2, self.conv2, self.bn2, self.out_shape2 = self.block(
            block_id=2, in_shape=self.out_shape1, channels_fact=4)

        self.up3, self.pd3, self.conv3, self.bn3, self.out_shape3 = self.block(
            block_id=3, in_shape=self.out_shape2, channels_fact=2)

        self.up4, self.pd4, self.conv4, self.bn4, self.out_shape4 = self.block(
            block_id=4, in_shape=self.out_shape3, channels_fact=1)

        block5_recon = self.end_block(block_id=5, in_shape=self.out_shape4)
        self.up5_r, self.pd5_r, self.conv5_r, self.end_up_r = block5_recon

        block5_scale = self.end_block(block_id=5, in_shape=self.out_shape4)
        self.up5_s, self.pd5_s, self.conv5_s, self.end_up_s = block5_scale

    def forward(self, z):
        """Forward pass of the decoder is to decode."""
        h1 = self.relu(self.l0(z))
        h1 = h1.view((-1,) + self.in_shape)
        h2 = self.leakyrelu(self.bn1(self.conv1(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn2(self.conv2(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn3(self.conv3(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn4(self.conv4(self.pd4(self.up4(h4)))))

        h6_r = self.conv5_r(self.pd5_r(self.up5_r(h5)))
        h6_s = self.conv5_s(self.pd5_s(self.up5_s(h5)))

        recon = self.end_up_r(h6_r)
        if self.with_sigmoid:
            recon = self.sigmoid(recon)
        scale_b = self.end_up_s(h6_s)
        return recon, scale_b


class VaeConvPlus(nn.Module):
    def __init__(self, latent_dim, img_shape, with_sigmoid):
        super(VaeConvPlus, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.with_sigmoid = with_sigmoid

        self.encoder = EncoderConvPlus(
            latent_dim=self.latent_dim,
            img_shape=self.img_shape)

        self.decoder = DecoderConvPlus(
            latent_dim=self.latent_dim,
            img_shape=self.img_shape,
            with_sigmoid=self.with_sigmoid,
            in_shape=self.encoder.enc5_out_shape)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparametrize(mu, logvar)
        res, scale_b = self.decoder(z)
        return res, scale_b, mu, logvar


class Discriminator(nn.Module):
    def dis_conv_output_size(self, in_shape, out_channels):
        return conv_output_size(
                in_shape, out_channels,
                kernel_size=DIS_KS,
                stride=DIS_STR,
                padding=DIS_PAD,
                dilation=DIS_DIL)

    def __init__(self, latent_dim, img_shape):
        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # discriminator
        self.dis1 = nn.Conv2d(
            in_channels=self.img_shape[0],
            out_channels=DIS_C,
            kernel_size=DIS_KS,
            stride=DIS_STR,
            padding=DIS_PAD)
        self.bn1 = nn.BatchNorm2d(self.dis1.out_channels)

        self.dis1_out_shape = self.dis_conv_output_size(
            in_shape=self.img_shape,
            out_channels=self.dis1.out_channels)

        self.dis2 = nn.Conv2d(
            in_channels=self.dis1.out_channels,
            out_channels=DIS_C * 2,
            kernel_size=DIS_KS,
            stride=DIS_STR,
            padding=DIS_PAD)
        self.bn2 = nn.BatchNorm2d(self.dis2.out_channels)
        self.dis2_out_shape = self.dis_conv_output_size(
            in_shape=self.dis1_out_shape,
            out_channels=self.dis2.out_channels)

        self.dis3 = nn.Conv2d(
            in_channels=self.dis2.out_channels,
            out_channels=DIS_C * 4,
            kernel_size=DIS_KS,
            stride=DIS_STR,
            padding=DIS_PAD)
        self.bn3 = nn.BatchNorm2d(self.dis3.out_channels)
        self.dis3_out_shape = self.dis_conv_output_size(
            in_shape=self.dis2_out_shape,
            out_channels=self.dis3.out_channels)

        self.dis4 = nn.Conv2d(
            in_channels=self.dis3.out_channels,
            out_channels=DIS_C * 8,
            kernel_size=DIS_KS,
            stride=DIS_STR,
            padding=DIS_PAD)
        self.bn4 = nn.BatchNorm2d(self.dis4.out_channels)
        self.dis4_out_shape = self.dis_conv_output_size(
            in_shape=self.dis3_out_shape,
            out_channels=self.dis4.out_channels)

        self.fcs_infeatures = functools.reduce(
            (lambda x, y: x * y), self.dis4_out_shape)

        # Two layers to generate mu and log sigma2 of Gaussian
        # Distribution of features
        self.fc1 = nn.Linear(
            in_features=self.fcs_infeatures,
            out_features=1)

        # TODO(nina): Add FC layers here to improve Discriminator

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
        h5 = h4.view(-1, self.fcs_infeatures)
        h5_feature = self.fc1(h5)
        prob = self.sigmoid(h5_feature)
        prob = prob.view(-1, 1)

        return prob, 0, 0  # h5_feature,  h5_logvar
