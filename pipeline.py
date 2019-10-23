"""Data processing pipeline."""

import functools
import jinja2
import logging
import luigi
import math
import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
import pickle
import random

import torch
import torch.autograd
from torch.nn import functional as F
import torch.optim
import torch.utils.data
import visdom

import datasets
import losses
import metrics
import nn
import train_utils

import warnings
warnings.filterwarnings("ignore")

# Decide on using segmentations, image intensities or fmri,
DATASET_NAME = 'cryo_exp'

HOME_DIR = '/scratch/users/nmiolane'
OUTPUT_DIR = os.path.join(HOME_DIR, 'output_%s' % DATASET_NAME)
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train_vae')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

DEBUG = False

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

IMG_SHAPE = (1, 128, 128)
DATA_DIM = functools.reduce((lambda x, y: x * y), IMG_SHAPE)
IMG_DIM = len(IMG_SHAPE)
LATENT_DIM = 3
NN_TYPE = 'conv_plus'
SPD = False
if SPD:
    NN_TYPE = 'conv_plus'
assert NN_TYPE in ['toy', 'fc', 'conv', 'conv_plus']

NN_ARCHITECTURE = {
    'img_shape': IMG_SHAPE,
    'data_dim': DATA_DIM,
    'latent_dim': LATENT_DIM,
    'nn_type': NN_TYPE,
    'with_sigmoid': True,
    'spd': SPD}

BATCH_SIZES = {15: 128, 25: 64, 64: 32, 90: 32, 96: 32, 100: 8, 128: 8}
BATCH_SIZE = BATCH_SIZES[IMG_SHAPE[1]]
FRAC_TEST = 0.1
FRAC_VAL = 0.2
N_SES_DEBUG = 3
if DEBUG:
    FRAC_VAL = 0.5
CKPT_PERIOD = 5

AXIS = {'fmri': 3, 'mri': 1, 'seg': 1}

PRINT_INTERVAL = 10
RECONSTRUCTIONS = ('bce_on_intensities', 'adversarial')
REGULARIZATIONS = ('kullbackleibler')
WEIGHTS_INIT = 'xavier'  #'custom'
REGU_FACTOR = 0.003

N_EPOCHS = 300
if DEBUG:
    N_EPOCHS = 2
    N_FILEPATHS = 10

LR = 15e-6
if 'adversarial' in RECONSTRUCTIONS:
    LR = 0.001  # 0.002 # 0.0002

TRAIN_PARAMS = {
    'lr': LR,
    'batch_size': BATCH_SIZE,
    'beta1': 0.5,
    'beta2': 0.999,
    'weights_init': WEIGHTS_INIT,
    'reconstructions': RECONSTRUCTIONS,
    'regularizations': REGULARIZATIONS
    }

NEURO_DIR = '/neuro'

LOADER = jinja2.FileSystemLoader('./templates/')
TEMPLATE_ENVIRONMENT = jinja2.Environment(
    autoescape=False,
    loader=LOADER)
TEMPLATE_NAME = 'report.jinja2'


class Train(luigi.Task):
    train_dir = TRAIN_DIR
    losses_path = os.path.join(train_dir, 'losses')
    train_losses_path = os.path.join(train_dir, 'train_losses.pkl')
    val_losses_path = os.path.join(train_dir, 'val_losses.pkl')

    def requires(self):
        pass

    def print_train_logs(self,
                         epoch,
                         batch_idx, n_batches, n_data, n_batch_data,
                         loss,
                         loss_reconstruction, loss_regularization,
                         loss_discriminator=0, loss_generator=0,
                         dx=0, dgex=0, dgz=0):

        loss = loss / n_batch_data
        loss_reconstruction = loss_reconstruction / n_batch_data
        loss_regularization = loss_regularization / n_batch_data
        loss_discriminator = loss_discriminator / n_batch_data
        loss_generator = loss_generator / n_batch_data

        string_base = ('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'
                       + '\nReconstruction: {:.6f}, Regularization: {:.6f}')

        if 'adversarial' in RECONSTRUCTIONS:
            string_base += (
                ', Discriminator: {:.6f}; Generator: {:.6f},'
                + 'D(x): {:.3f}, D(G(E(x))): {:.3f}, D(G(z)): {:.3f}')

        if 'adversarial' not in RECONSTRUCTIONS:
            logging.info(
                string_base.format(
                    epoch, batch_idx * n_batch_data, n_data,
                    100. * batch_idx / n_batches,
                    loss, loss_reconstruction, loss_regularization))
        else:
            logging.info(
                string_base.format(
                    epoch, batch_idx * n_batch_data, n_data,
                    100. * batch_idx / n_batches,
                    loss, loss_reconstruction, loss_regularization,
                    loss_discriminator, loss_generator,
                    dx, dgex, dgz))

    def train(self, epoch, train_loader,
              modules, optimizers,
              reconstructions=RECONSTRUCTIONS,
              regularizations=REGULARIZATIONS):
        """
        - modules: a dict with the bricks of the model,
        eg. encoder, decoder, discriminator, depending on the architecture
        - optimizers: a dict with optimizers corresponding to each module.
        """
        train_vis = visdom.Visdom()
        train_vis.env = 'train_images'
        data_win = None
        recon_win = None
        from_prior_win = None

        for module in modules.values():
            module.train()

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in RECONSTRUCTIONS:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)

        for batch_idx, batch_data in enumerate(train_loader):
            if DEBUG:
                if batch_idx < n_batches - 3:
                    continue
            if DATASET_NAME not in ['cryo']:
                batch_data = batch_data[0].to(DEVICE)
            else:
                batch_data = batch_data.to(DEVICE).float()
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                mu, logvar).to(DEVICE)
            batch_recon, scale_b = decoder(z)

            z_from_prior = nn.sample_from_prior(
                LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, scale_b_from_prior = decoder(
                z_from_prior)

            if 'adversarial' in reconstructions:
                # From:
                # Autoencoding beyond pixels using a learned similarity metric
                # arXiv:1512.09300v2
                discriminator = modules['discriminator_reconstruction']
                real_labels = torch.full((n_batch_data,), 1, device=DEVICE)
                fake_labels = torch.full((n_batch_data,), 0, device=DEVICE)

                # -- Update DiscriminatorGan
                labels_data, h_data, _ = discriminator(
                    batch_data)
                labels_recon, h_recon, h_logvar_recon = discriminator(
                    batch_recon.detach())
                labels_from_prior, _, _ = discriminator(
                    batch_from_prior.detach())

                loss_dis_data = F.binary_cross_entropy(
                    labels_data,
                    real_labels)
                loss_dis_recon = F.binary_cross_entropy(
                    labels_recon,
                    fake_labels)
                loss_dis_from_prior = F.binary_cross_entropy(
                    labels_from_prior,
                    fake_labels)

                # TODO(nina): add loss_dis_recon
                loss_discriminator = (
                    loss_dis_data
                    + loss_dis_from_prior)

                # Fill gradients on discriminator only
                loss_discriminator.backward(retain_graph=True)

                # Need to do optimizer step here, as gradients
                # of the reconstruction with discriminator features
                # may fill the discriminator's weights and we do not
                # update the discriminator with the reconstruction loss.
                optimizers['discriminator_reconstruction'].step()

                # -- Update Generator/DecoderGAN
                # Note that we need to do a forward pass with detached vars
                # in order not to propagate gradients through the encoder
                batch_recon_detached, _ = decoder(z.detach())
                # Note that we don't need to do it for batch_from_prior
                # as it doesn't come from the encoder

                labels_recon, _, _ = discriminator(
                    batch_recon_detached)
                labels_from_prior, _, _ = discriminator(
                    batch_from_prior)

                loss_generator_recon = F.binary_cross_entropy(
                    labels_recon,
                    real_labels)

                # TODO(nina): add loss_generator_from_prior
                loss_generator = loss_generator_recon

                # Fill gradients on generator only
                loss_generator.backward()

            if 'mse_on_intensities' in reconstructions:
                loss_reconstruction = losses.mse_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'bce_on_intensities' in reconstructions:
                loss_reconstruction = losses.bce_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'mse_on_features' in reconstructions:
                # TODO(nina): Investigate stat interpretation
                # of using the logvar from the recon
                loss_reconstruction = losses.mse_on_features(
                    h_recon, h_data, h_logvar_recon)
                # Fill gradients on encoder and generator
                # but not on discriminator
                loss_reconstruction.backward(retain_graph=True)

            if 'kullbackleibler' in regularizations:
                loss_regularization = losses.kullback_leibler(mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            if 'kullbackleibler_circle' in regularizations:
                loss_regularization = losses.kullback_leibler_circle(
                        mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            if 'on_circle' in regularizations:
                loss_regularization = losses.on_circle(
                        mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            if 'adversarial' in regularizations:
                # From: Adversarial autoencoders
                # https://arxiv.org/pdf/1511.05644.pdf
                discriminator = modules['discriminator_regularization']
                raise NotImplementedError(
                    'Adversarial regularization not implemented.')

            if 'wasserstein' in regularizations:
                raise NotImplementedError(
                    'Wasserstein regularization not implemented.')

            optimizers['encoder'].step()
            optimizers['decoder'].step()

            loss = loss_reconstruction + loss_regularization
            if 'adversarial' in RECONSTRUCTIONS:
                loss += loss_discriminator + loss_generator

            if batch_idx % PRINT_INTERVAL == 0:
                # TODO(nina): Why didn't we need .mean() on 64x64?
                if 'adversarial' in RECONSTRUCTIONS:
                    self.print_train_logs(
                        epoch,
                        batch_idx, n_batches, n_data, n_batch_data,
                        loss, loss_reconstruction, loss_regularization,
                        loss_discriminator, loss_generator,
                        labels_data.mean(),
                        labels_recon.mean(),
                        labels_from_prior.mean())
                else:
                    self.print_train_logs(
                        epoch,
                        batch_idx, n_batches, n_data, n_batch_data,
                        loss, loss_reconstruction, loss_regularization)

                # Visdom first images of batch
                # TODO(nina): Why does it print black images for batch_data??
                # print(torch.sum(batch_data[0]))
                height = 150 * IMG_SHAPE[1] / 64
                width = 150 * IMG_SHAPE[2] / 64
                data_win = train_vis.image(
                    batch_data[0],
                    win=data_win,
                    opts=dict(
                        title='Train Epoch {}: Data'.format(epoch),
                        height=height, width=width))
                recon_win = train_vis.image(
                    batch_recon[0],
                    win=recon_win,
                    opts=dict(
                        title='Train Epoch {}: Reconstruction'.format(epoch),
                        height=height, width=width))
                from_prior_win = train_vis.image(
                    batch_from_prior[0],
                    win=from_prior_win,
                    opts=dict(
                        title='Train Epoch {}: From prior'.format(epoch),
                        height=height, width=width))

            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_regularization += loss_regularization.item()
            if 'adversarial' in RECONSTRUCTIONS:
                total_loss_discriminator += loss_discriminator.item()
                total_loss_generator += loss_generator.item()
            total_loss += loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        if 'adversarial' in RECONSTRUCTIONS:
            average_loss_discriminator = total_loss_discriminator / n_data
            average_loss_generator = total_loss_generator / n_data
        average_loss = total_loss / n_data

        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, average_loss))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        if 'adversarial' in RECONSTRUCTIONS:
            train_losses['discriminator'] = average_loss_discriminator
            train_losses['generator'] = average_loss_generator
        train_losses['total'] = average_loss
        return train_losses

    def val(self, epoch, val_loader, modules,
            reconstructions=RECONSTRUCTIONS,
            regularizations=REGULARIZATIONS):

        vis = visdom.Visdom()
        vis.env = 'val_images'
        data_win = None
        recon_win = None
        from_prior_win = None

        for module in modules.values():
            module.eval()

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in RECONSTRUCTIONS:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                if DEBUG:
                    if batch_idx < n_batches - 3:
                        continue
                if DATASET_NAME not in ['cryo', 'connectomes']:
                    batch_data = batch_data[0].to(DEVICE)
                else:
                    batch_data = batch_data.to(DEVICE).float()
                n_batch_data = batch_data.shape[0]

                encoder = modules['encoder']
                decoder = modules['decoder']

                mu, logvar = encoder(batch_data)
                z = nn.sample_from_q(mu, logvar).to(DEVICE)
                batch_recon, scale_b = decoder(z)

                z_from_prior = nn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
                batch_from_prior, scale_b_from_prior = decoder(
                    z_from_prior)

                if 'adversarial' in reconstructions:
                    # From:
                    # Autoencoding beyond pixels using a learned
                    # similarity metric
                    # arXiv:1512.09300v2
                    discriminator = modules['discriminator_reconstruction']
                    real_labels = torch.full((n_batch_data,), 1, device=DEVICE)
                    fake_labels = torch.full((n_batch_data,), 0, device=DEVICE)

                    # -- Compute DiscriminatorGan Loss
                    labels_data, h_data, _ = discriminator(batch_data)
                    labels_recon, h_recon, h_logvar_recon = discriminator(
                        batch_recon.detach())
                    labels_from_prior, _, _ = discriminator(
                        batch_from_prior.detach())

                    loss_dis_data = F.binary_cross_entropy(
                        labels_data,
                        real_labels)
                    loss_dis_recon = F.binary_cross_entropy(
                        labels_recon,
                        fake_labels)
                    loss_dis_from_prior = F.binary_cross_entropy(
                        labels_from_prior,
                        fake_labels)

                    # TODO(nina): add loss_dis_recon
                    loss_discriminator = (
                        loss_dis_data
                        + loss_dis_from_prior)

                    # -- Compute Generator/DecoderGAN Loss
                    # Note that we need to do a forward pass with detached vars
                    # in order not to propagate gradients through the encoder
                    batch_recon_detached, _ = decoder(z.detach())
                    # Note that we don't need to do it for
                    # batch_from_prior
                    # as it doesn't come from the encoder

                    labels_recon, _, _ = discriminator(
                        batch_recon_detached)
                    labels_from_prior, _, _ = discriminator(
                        batch_from_prior)

                    loss_generator_recon = F.binary_cross_entropy(
                        labels_recon,
                        real_labels)

                    # TODO(nina): add loss_generator_from_prior
                    loss_generator = loss_generator_recon

                if 'mse_on_intensities' in reconstructions:
                    loss_reconstruction = losses.mse_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'bce_on_intensities' in reconstructions:
                    loss_reconstruction = losses.bce_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'mse_on_features' in reconstructions:
                    # TODO(nina): Investigate stat interpretation
                    # of using the logvar from the recon
                    loss_reconstruction = losses.mse_on_features(
                        h_recon, h_data, h_logvar_recon)

                if 'kullbackleibler' in regularizations:
                    loss_regularization = losses.kullback_leibler(
                        mu, logvar)

                if 'kullbackleibler_circle' in regularizations:
                    loss_regularization = losses.kullback_leibler_circle(
                            mu, logvar)

                if 'on_circle' in regularizations:
                    loss_regularization = losses.on_circle(
                            mu, logvar)

                if 'adversarial' in regularizations:
                    # From: Adversarial autoencoders
                    # https://arxiv.org/pdf/1511.05644.pdf
                    discriminator = modules['discriminator_regularization']
                    raise NotImplementedError(
                        'Adversarial regularization not implemented.')

                if 'wasserstein' in regularizations:
                    raise NotImplementedError(
                        'Wasserstein regularization not implemented.')

                loss = loss_reconstruction + loss_regularization
                if 'adversarial' in RECONSTRUCTIONS:
                    loss += loss_discriminator + loss_generator

                total_loss_reconstruction += loss_reconstruction.item()
                total_loss_regularization += loss_regularization.item()
                if 'adversarial' in RECONSTRUCTIONS:
                    total_loss_discriminator += loss_discriminator.item()
                    total_loss_generator += loss_generator.item()
                total_loss += loss.item()

                if batch_idx == n_batches - 1:
                    # On last batch: visualize
                    batch_data = batch_data.cpu().numpy()
                    batch_recon = batch_recon.cpu().numpy()
                    batch_from_prior = batch_from_prior.cpu().numpy()

                    # Visdom first images of last batch
                    height = 150 * IMG_SHAPE[1] / 64
                    width = 150 * IMG_SHAPE[2] / 64
                    data_win = vis.image(
                        batch_data[0][0]+0.5,
                        win=data_win,
                        opts=dict(
                            title='Val Epoch {}: Data'.format(epoch),
                            height=height, width=width))
                    recon_win = vis.image(
                        batch_recon[0][0],
                        win=recon_win,
                        opts=dict(
                            title='Val Epoch {}: Reconstruction'.format(
                                epoch),
                            height=height, width=width))
                    from_prior_win = vis.image(
                        batch_from_prior[0][0],
                        win=from_prior_win,
                        opts=dict(
                            title='Val Epoch {}: From prior'.format(epoch),
                            height=height, width=width))

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        if 'adversarial' in RECONSTRUCTIONS:
            average_loss_discriminator = total_loss_discriminator / n_data
            average_loss_generator = total_loss_generator / n_data
        average_loss = total_loss / n_data
        print('====> Val set loss: {:.4f}'.format(average_loss))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        if 'adversarial' in RECONSTRUCTIONS:
            val_losses['discriminator'] = average_loss_discriminator
            val_losses['generator'] = average_loss_generator
        val_losses['total'] = average_loss
        return val_losses

    def run(self):
        for directory in (self.train_dir, self.losses_path):
            if not os.path.isdir(directory):
                os.mkdir(directory)
                os.chmod(directory, 0o777)

        train_dataset, val_dataset = datasets.get_datasets(
                dataset_name=DATASET_NAME,
                frac_val=FRAC_VAL,
                batch_size=BATCH_SIZE,
                img_shape=IMG_SHAPE)

        train = torch.Tensor(train_dataset)
        val = torch.Tensor(val_dataset)

        logging.info('-- Train tensor: (%d, %d, %d, %d)' % train.shape)
        logging.info('-- Val tensor: (%d, %d, %d, %d)' % val.shape)

        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)
        val_dataset = torch.utils.data.TensorDataset(val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        vae = nn.VaeConvPlus(
            latent_dim=LATENT_DIM,
            img_shape=IMG_SHAPE,
            with_sigmoid=True).to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

        if 'adversarial' in RECONSTRUCTIONS:
            discriminator = nn.Discriminator(
                latent_dim=LATENT_DIM,
                img_shape=IMG_SHAPE).to(DEVICE)
            modules['discriminator_reconstruction'] = discriminator

        if 'adversarial' in REGULARIZATIONS:
            discriminator = nn.Discriminator(
                latent_dim=LATENT_DIM,
                img_shape=IMG_SHAPE).to(DEVICE)
            modules['discriminator_regularization'] = discriminator

        optimizers = {}
        optimizers['encoder'] = torch.optim.Adam(
            modules['encoder'].parameters(), lr=LR)
        optimizers['decoder'] = torch.optim.Adam(
            modules['decoder'].parameters(),
            lr=LR,
            betas=(TRAIN_PARAMS['beta1'], TRAIN_PARAMS['beta2']))

        if 'adversarial' in RECONSTRUCTIONS:
            optimizers['discriminator_reconstruction'] = torch.optim.Adam(
                modules['discriminator_reconstruction'].parameters(),
                lr=LR,
                betas=(TRAIN_PARAMS['beta1'], TRAIN_PARAMS['beta2']))

        if 'adversarial' in REGULARIZATIONS:
            optimizers['discriminator_regularization'] = torch.optim.Adam(
                modules['discriminator_regularization'].parameters(),
                lr=LR,
                betas=(TRAIN_PARAMS['beta1'], TRAIN_PARAMS['beta2']))

        for module in modules.values():
            if WEIGHTS_INIT == 'xavier':
                module.apply(train_utils.init_xavier_normal)
            elif WEIGHTS_INIT == 'kaiming':
                module.apply(train_utils.init_kaiming_normal)
            elif WEIGHTS_INIT == 'custom':
                module.apply(train_utils.init_custom)
            else:
                raise NotImplementedError(
                    'This weight initialization is not implemented.')

        vis2 = visdom.Visdom()
        vis2.env = 'losses'
        train_loss_window = vis2.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Epochs',
                      ylabel='Train loss',
                      title='Train loss',
                      legend=['loss']))
        val_loss_window = vis2.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Epochs',
                      ylabel='Val loss',
                      title='Val loss',
                      legend=['loss']))

        m, o, s, t, v = train_utils.init_training(
            self.train_dir, NN_ARCHITECTURE, TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v
        for epoch in range(start_epoch, N_EPOCHS):
            train_losses = self.train(
                epoch, train_loader, modules, optimizers,
                RECONSTRUCTIONS, REGULARIZATIONS)
            val_losses = self.val(
                epoch, val_loader, modules,
                RECONSTRUCTIONS, REGULARIZATIONS)

            train_losses_all_epochs.append(train_losses)
            val_losses_all_epochs.append(val_losses)

            # TODO(nina): Fix bug that losses do not show on visdom.
            train_loss = train_losses['total']
            val_loss = val_losses['total']
            vis2.line(
                X=torch.ones((1, 1)).cpu()*epoch,
                Y=torch.Tensor([train_loss]).unsqueeze(0).cpu(),
                win=train_loss_window,
                update='append')
            vis2.line(
                X=torch.ones((1, 1)).cpu()*epoch,
                Y=torch.Tensor([val_loss]).unsqueeze(0).cpu(),
                win=val_loss_window,
                update='append')

            if epoch % CKPT_PERIOD == 0:
                train_utils.save_checkpoint(
                    epoch=epoch, modules=modules, optimizers=optimizers,
                    dir_path=self.train_dir,
                    train_losses_all_epochs=train_losses_all_epochs,
                    val_losses_all_epochs=val_losses_all_epochs,
                    nn_architecture=NN_ARCHITECTURE,
                    train_params=TRAIN_PARAMS)

        for module_name, module in modules.items():
            module_path = os.path.join(
                self.train_dir, '{}.pth'.format(module_name))
            torch.save(module, module_path)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses_all_epochs, pkl)
        with open(self.output()['val_losses'].path, 'wb') as pkl:
            pickle.dump(val_losses_all_epochs, pkl)

    def output(self):
        return {'train_losses': luigi.LocalTarget(self.train_losses_path),
                'val_losses': luigi.LocalTarget(self.val_losses_path)}


class Report(luigi.Task):
    report_path = os.path.join(REPORT_DIR, 'report.html')

    def requires(self):
        return Train()

    def get_last_epoch(self):
        # Placeholder
        epoch_id = N_EPOCHS - 1
        return epoch_id

    def get_loss_history(self):
        last_epoch = self.get_last_epoch()
        loss_history = []
        for epoch_id in range(last_epoch):
            path = os.path.join(
                TRAIN_DIR, 'losses', 'epoch_%d' % epoch_id)
            loss = np.load(path)
            loss_history.append(loss)
        return loss_history

    def load_data(self, epoch_id):
        data_path = os.path.join(
            TRAIN_DIR, 'imgs', 'epoch_%d_data.npy' % epoch_id)
        data = np.load(data_path)
        return data

    def load_recon(self, epoch_id):
        recon_path = os.path.join(
            TRAIN_DIR, 'imgs', 'epoch_%d_recon.npy' % epoch_id)
        recon = np.load(recon_path)
        return recon

    def load_from_prior(self, epoch_id):
        from_prior_path = os.path.join(
            TRAIN_DIR, 'imgs', 'epoch_%d_from_prior.npy' % epoch_id)
        from_prior = np.load(from_prior_path)
        return from_prior

    def plot_losses(self, loss_types=['total']):
        last_epoch = self.get_last_epoch()
        epochs = range(last_epoch+1)

        loss_types = [
            'total',
            'discriminator', 'generator',
            'reconstruction', 'regularization']
        train_losses = {loss_type: [] for loss_type in loss_types}
        val_losses = {loss_type: [] for loss_type in loss_types}

        for i in epochs:
            path = os.path.join(TRAIN_DIR, 'losses', 'epoch_%d.pkl' % i)
            train_val = pickle.load(open(path, 'rb'))
            train = train_val['train']
            val = train_val['val']

            for loss_type in loss_types:
                loss = train[loss_type]
                train_losses[loss_type].append(loss)

                loss = val[loss_type]
                val_losses[loss_type].append(loss)

        # Total
        params = {
            'legend.fontsize': 'xx-large',
            'figure.figsize': (40, 60),
            'axes.labelsize': 'xx-large',
            'axes.titlesize': 'xx-large',
            'xtick.labelsize': 'xx-large',
            'ytick.labelsize': 'xx-large'}
        pylab.rcParams.update(params)

        n_rows = 3
        n_cols = 2
        fig = plt.figure(figsize=(20, 22))

        # Total
        plt.subplot(n_rows, n_cols, 1)
        plt.plot(train_losses['total'])
        plt.title('Train Loss')

        plt.subplot(n_rows, n_cols, 2)
        plt.plot(val_losses['total'])
        plt.title('Val Loss')

        # Decomposed in sublosses

        plt.subplot(n_rows, n_cols, 3)
        plt.plot(epochs, train_losses['discriminator'])
        plt.plot(epochs, train_losses['generator'])
        plt.plot(epochs, train_losses['reconstruction'])
        plt.plot(epochs, train_losses['regularization'])
        plt.title('Train Loss Decomposed')
        plt.legend(
            [loss_type for loss_type in loss_types if loss_type != 'total'],
            loc='upper right')

        plt.subplot(n_rows, n_cols, 4)
        plt.plot(epochs, val_losses['discriminator'])
        plt.plot(epochs, val_losses['generator'])
        plt.plot(epochs, val_losses['reconstruction'])
        plt.plot(epochs, val_losses['regularization'])
        plt.title('Val Loss Decomposed')
        plt.legend(
            [loss_type for loss_type in loss_types if loss_type != 'total'],
            loc='upper right')

        # Only DiscriminatorGan and Generator
        plt.subplot(n_rows, n_cols, 5)
        plt.plot(epochs, train_losses['discriminator'])
        plt.plot(epochs, train_losses['generator'])
        plt.title('Train Loss: Discriminator and Generator only')
        plt.legend(
            [loss_type for loss_type in loss_types
             if (loss_type == 'discriminator'
                 or loss_type == 'generator')],
            loc='upper right')

        plt.subplot(n_rows, n_cols, 6)
        plt.plot(epochs, val_losses['discriminator'])
        plt.plot(epochs, val_losses['generator'])
        plt.title('Val Loss: Discriminator and Generator only')
        plt.legend(
            [loss_type for loss_type in loss_types
             if (loss_type == 'discriminator'
                 or loss_type == 'generator')],
            loc='upper right')

        fname = os.path.join(REPORT_DIR, 'losses.png')
        fig.savefig(fname, pad_inches=1.)
        plt.clf()
        return os.path.basename(fname)

    def plot_images(self,
                    n_plot_epochs=4,
                    n_imgs_per_epoch=12):
        last_epoch = self.get_last_epoch()
        n_plot_epochs = math.gcd(n_plot_epochs-1, last_epoch) + 1
        n_imgs_type = 3
        by = int(last_epoch / (n_plot_epochs - 1))

        n_rows = n_imgs_type * n_plot_epochs
        n_cols = n_imgs_per_epoch
        fig = plt.figure(figsize=(20*n_cols, 20*n_rows))

        for epoch_id in range(0, last_epoch+1, by):
            data = self.load_data(epoch_id)
            recon = self.load_recon(epoch_id)
            from_prior = self.load_from_prior(epoch_id)

            for img_id in range(n_imgs_per_epoch):
                subplot_epoch_id = (
                    epoch_id * n_imgs_type * n_imgs_per_epoch / by)

                subplot_id = subplot_epoch_id + img_id + 1
                plt.subplot(n_rows, n_cols, subplot_id)
                plt.imshow(data[img_id][0], cmap='gray')
                plt.axis('off')

                subplot_id = subplot_epoch_id + n_imgs_per_epoch + img_id + 1
                plt.subplot(n_rows, n_cols, subplot_id)
                plt.imshow(recon[img_id][0], cmap='gray')
                plt.axis('off')

                subplot_id = (
                    subplot_epoch_id + 2 * n_imgs_per_epoch + img_id + 1)
                plt.subplot(n_rows, n_cols, subplot_id)
                plt.imshow(from_prior[img_id][0], cmap='gray')
                plt.axis('off')

                plt.tight_layout()

        fname = os.path.join(REPORT_DIR, 'images.png')
        fig.savefig(fname, pad_inches=1.)
        plt.clf()
        return os.path.basename(fname)

    def compute_reconstruction_metrics(self):
        epoch_id = self.get_last_epoch()
        data = self.load_data(epoch_id)
        recon = self.load_recon(epoch_id)

        # TODO(nina): Rewrite mi and fid in pytorch
        mutual_information = metrics.mutual_information(recon, data)
        fid = metrics.frechet_inception_distance(recon, data)

        data = torch.Tensor(data)
        recon = torch.Tensor(recon)

        bce = metrics.binary_cross_entropy(recon, data)
        mse = metrics.mse(recon, data)
        l1_norm = metrics.l1_norm(recon, data)

        context = {
            'title': 'Vaetree Report',
            'bce': bce,
            'mse': mse,
            'l1_norm': l1_norm,
            'mutual_information': mutual_information,
            'fid': fid,
            }
        # Placeholder
        return context

    def run(self):
        # Loss functions
        loss_basename = self.plot_losses()

        # Sample of Data, Reconstruction, Generated from Prior
        imgs_basename = self.plot_images()

        # Table of Reconstruction Metrics
        context = self.compute_reconstruction_metrics()

        # PCA on latent space - 5 first components

        # K-means on latent space - 4 clusters

        with open(self.output().path, 'w') as f:
            template = TEMPLATE_ENVIRONMENT.get_template(TEMPLATE_NAME)
            html = template.render(context)
            f.write(html)

    def output(self):
        return luigi.LocalTarget(self.report_path)


class RunAll(luigi.Task):
    def requires(self):
        return Report()

    def output(self):
        return luigi.LocalTarget('dummy')


def init():
    for directory in [OUTPUT_DIR, TRAIN_DIR, REPORT_DIR]:
        if not os.path.isdir(directory):
            os.mkdir(directory)
            os.chmod(directory, 0o777)

    logging.basicConfig(level=logging.INFO)
    logging.info('start')
    luigi.run(
        main_task_cls=RunAll(),
        cmdline_args=[
            '--local-scheduler',
        ])


if __name__ == "__main__":
    init()
