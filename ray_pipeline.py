"""Data processing pipeline."""

import functools
import logging
import numpy as np
import os
import random

import ray
from ray import tune

from ray.tune import Trainable
from ray.tune.schedulers import HyperBandScheduler


import torch
import torch.autograd
from torch.nn import functional as F
import torch.optim
import torch.utils.data
import visdom

import datasets
import losses
import nn
import train_utils

import warnings
warnings.filterwarnings("ignore")

DEBUG = False

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

REDIS_ADDRESS = '127.0.0.1:6379'

# Seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATASET_NAME = 'connectomes'

HOME_DIR = '/scratch/users/nmiolane'
OUTPUT_DIR = os.path.join(HOME_DIR, 'output_%s' % DATASET_NAME)
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train_vae')

IMG_SHAPE = (1, 100, 100)
DATA_DIM = functools.reduce((lambda x, y: x * y), IMG_SHAPE)
IMG_DIM = len(IMG_SHAPE)
LATENT_DIM = 3
NN_TYPE = 'gan'
SPD = False
if SPD:
    NN_TYPE = 'conv'
assert NN_TYPE == 'gan'

NN_ARCHITECTURE = {
    'img_shape': IMG_SHAPE,
    'data_dim': DATA_DIM,
    'latent_dim': LATENT_DIM,
    'nn_type': NN_TYPE,
    'spd': SPD
    }


BATCH_SIZES = {15: 128, 25: 64, 64: 32, 96: 32, 100: 8, 128: 8}
BATCH_SIZE = BATCH_SIZES[IMG_SHAPE[1]]
FRAC_TEST = 0.1
FRAC_VAL = 0.2

N_SES_DEBUG = 3
if DEBUG:
    FRAC_VAL = 0.5
CKPT_PERIOD = 5

CMAPS = {'connectomes': 'Spectral'}
CMAP = CMAPS[DATASET_NAME]

AXIS = {'fmri': 3, 'mri': 1, 'seg': 1}

PRINT_INTERVAL = 10
torch.backends.cudnn.benchmark = True

# From: Autoencoding beyond pixels using a learned similarity metric
# arXiv:1512.09300v2
RECONSTRUCTIONS = ('bce_on_intensities', 'adversarial')
# TODO(nina): Consider implementing:
# - adversarial regularization: https://arxiv.org/pdf/1511.05644.pdf
# - wasserstein regularization
REGULARIZATIONS = ('kullbackleibler',)
WEIGHTS_INIT = 'custom'
REGU_FACTOR = 0.003

N_EPOCHS = 61
if DEBUG:
    N_EPOCHS = 2
    N_FILEPATHS = 10

LR = 15e-6
if 'adversarial' in RECONSTRUCTIONS:
    LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
N_EXPERIMENTS = 1


TRAIN_PARAMS = {
    'lr': LR,
    'beta1': BETA1,
    'beta2': BETA2,
    'weights_init': WEIGHTS_INIT,
    'reconstructions': RECONSTRUCTIONS,
    'regularizations': REGULARIZATIONS
    }


class Train(Trainable):

    def _setup(self, config):
        train_loader, val_loader = datasets.get_loaders(
                dataset_name=DATASET_NAME,
                frac_val=FRAC_VAL,
                batch_size=BATCH_SIZE,
                img_shape=IMG_SHAPE)

        m, o, s, t, v = train_utils.init_training(
            train_dir=TRAIN_DIR,
            nn_architecture=NN_ARCHITECTURE,
            train_params=TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.modules = modules
        self.optimizers = optimizers
        self.start_epoch = start_epoch

        self.train_losses_all_epochs = train_losses_all_epochs
        self.val_losses_all_epochs = val_losses_all_epochs

    def _train_iteration(self):
        """
        One epoch.
        - modules: a dict with the bricks of the model,
        eg. encoder, decoder, discriminator, depending on the architecture
        - optimizers: a dict with optimizers corresponding to each module.
        """
        epoch = self._iteration
        for module in self.modules.values():
            module.train()

        train_vis = visdom.Visdom()
        train_vis.env = 'train_images'
        data_win = None
        recon_win = None
        from_prior_win = None

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in RECONSTRUCTIONS:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(self.train_loader.dataset)
        n_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            if DEBUG:
                if batch_idx < n_batches - 3:
                    continue
            if DATASET_NAME not in ['cryo', 'cryo_sim',
                                    'cryo_exp', 'connectomes']:
                batch_data = batch_data[0].to(DEVICE)
            else:
                batch_data = batch_data.to(DEVICE).float()
            n_batch_data = len(batch_data)

            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            encoder = self.modules['encoder']
            decoder = self.modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                mu, logvar).to(DEVICE)
            batch_recon, scale_b = decoder(z)

            z_from_prior = nn.sample_from_prior(
                LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, scale_b_from_prior = decoder(
                z_from_prior)

            if 'adversarial' in RECONSTRUCTIONS:
                # From:
                # Autoencoding beyond pixels using a learned similarity metric
                # arXiv:1512.09300v2
                discriminator = self.modules['discriminator_reconstruction']
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
                loss_dis_from_prior = F.binary_cross_entropy(
                    labels_from_prior,
                    fake_labels)

                # TODO(nina): add loss_dis_recon
                # loss_dis_recon = F.binary_cross_entropy(
                #   labels_recon,
                #    fake_labels)
                loss_discriminator = (
                    loss_dis_data
                    + loss_dis_from_prior)

                # Fill gradients on discriminator only
                loss_discriminator.backward(retain_graph=True)

                # Need to do optimizer step here, as gradients
                # of the reconstruction with discriminator features
                # may fill the discriminator's weights and we do not
                # update the discriminator with the reconstruction loss.
                self.optimizers['discriminator_reconstruction'].step()

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

            if 'mse_on_intensities' in RECONSTRUCTIONS:
                loss_reconstruction = losses.mse_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'bce_on_intensities' in RECONSTRUCTIONS:
                loss_reconstruction = losses.bce_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'mse_on_features' in RECONSTRUCTIONS:
                # TODO(nina): Investigate stat interpretation
                # of using the logvar from the recon
                loss_reconstruction = losses.mse_on_features(
                    h_recon, h_data, h_logvar_recon)
                # Fill gradients on encoder and generator
                # but not on discriminator
                loss_reconstruction.backward(retain_graph=True)

            if 'kullbackleibler' in REGULARIZATIONS:
                loss_regularization = losses.kullback_leibler(mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            self.optimizers['encoder'].step()
            self.optimizers['decoder'].step()

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

                data_win = train_vis.heatmap(
                    batch_data[0, 0],
                    win=data_win,
                    opts=dict(
                        title='Train Epoch {}: Data'.format(epoch),
                        height=height,
                        width=width + 1/6 * width,
                        colormap=CMAP))
                recon_win = train_vis.heatmap(
                    batch_recon[0, 0],
                    win=recon_win,
                    opts=dict(
                        title='Train Epoch {}: Reconstruction'.format(epoch),
                        height=height,
                        width=width + 1/6 * width,
                        colormap=CMAP))
                from_prior_win = train_vis.heatmap(
                    batch_from_prior[0, 0],
                    win=from_prior_win,
                    opts=dict(
                        title='Train Epoch {}: From prior'.format(epoch),
                        height=height,
                        width=width + 1/6 * width,
                        colormap=CMAP))

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

        self.train_losses_all_epochs.append(train_losses)

        if epoch % CKPT_PERIOD == 0:
            train_utils.save_checkpoint(
                dir_path=TRAIN_DIR,
                nn_architecture=NN_ARCHITECTURE,
                epoch=epoch,
                modules=self.modules,
                optimizers=self.optimizers,
                train_losses_all_epochs=self.train_losses_all_epochs,
                val_losses_all_epochs=self.val_losses_all_epochs)

    def _train(self):
        self._train_iteration()
        return self._test()

    def _test(self):
        epoch = self._iteration
        for module in self.modules.values():
            module.eval()

        vis = visdom.Visdom()
        vis.env = 'val_images'
        data_win = None
        recon_win = None
        from_prior_win = None

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in RECONSTRUCTIONS:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(self.val_loader.dataset)
        n_batches = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                if DEBUG:
                    if batch_idx < n_batches - 3:
                        continue
                if DATASET_NAME not in ['cryo', 'cryo_sim',
                                        'cryo_exp', 'connectomes']:
                    batch_data = batch_data[0].to(DEVICE)
                else:
                    batch_data = batch_data.to(DEVICE).float()
                n_batch_data = batch_data.shape[0]

                encoder = self.modules['encoder']
                decoder = self.modules['decoder']

                mu, logvar = encoder(batch_data)
                z = nn.sample_from_q(mu, logvar).to(DEVICE)
                batch_recon, scale_b = decoder(z)

                z_from_prior = nn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
                batch_from_prior, scale_b_from_prior = decoder(
                    z_from_prior)

                if 'adversarial' in RECONSTRUCTIONS:
                    discriminator = self.modules[
                        'discriminator_reconstruction']
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
                    loss_dis_from_prior = F.binary_cross_entropy(
                        labels_from_prior,
                        fake_labels)

                    # TODO(nina): add loss_dis_recon
                    # loss_dis_recon = F.binary_cross_entropy(
                    #    labels_recon,
                    #    fake_labels)
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

                if 'mse_on_intensities' in RECONSTRUCTIONS:
                    loss_reconstruction = losses.mse_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'bce_on_intensities' in RECONSTRUCTIONS:
                    loss_reconstruction = losses.bce_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'mse_on_features' in RECONSTRUCTIONS:
                    # TODO(nina): Investigate stat interpretation
                    # of using the logvar from the recon
                    loss_reconstruction = losses.mse_on_features(
                        h_recon, h_data, h_logvar_recon)

                if 'kullbackleibler' in REGULARIZATIONS:
                    loss_regularization = losses.kullback_leibler(
                        mu, logvar)

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

        self.val_losses_all_epochs.append(val_losses)
        return {'average_loss': average_loss}

    def _save(self, checkpoint_dir=TRAIN_DIR):
        epoch = self._iteration
        train_utils.save_checkpoint(
            dir_path=checkpoint_dir,
            nn_architecture=NN_ARCHITECTURE,
            epoch=epoch,
            modules=self.modules,
            optimizers=self.optimizers,
            train_losses_all_epochs=self.train_losses_all_epochs,
            val_losses_all_epochs=self.val_losses_all_epochs)
        checkpoint_path = os.path.join(
            checkpoint_dir, 'epoch_%d_checkpoint.pth' % epoch)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        epoch_id = None  # HACK: restore last one
        train_dir = os.path.dirname(checkpoint_path)
        output = os.path.dirname(train_dir)
        for module_name in self.modules.keys():
            self.modules[module_name] = train_utils.load_module(
                output=output,
                algo_name='vae',
                module_name=module_name,
                epoch_id=epoch_id)

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


def init():
    for directory in [OUTPUT_DIR, TRAIN_DIR]:
        if not os.path.isdir(directory):
            os.mkdir(directory)
            os.chmod(directory, 0o777)

    logging.basicConfig(level=logging.INFO)
    logging.info('start')


if __name__ == "__main__":
    init()

    ray.init()  # redis_address=REDIS_ADDRESS, head=True)

    sched = HyperBandScheduler(
        time_attr='training_iteration', metric='average_loss', mode='min')
    tune.run(
        Train,
        scheduler=sched,
        **{
            'stop': {
                'training_iteration': N_EPOCHS,
            },
            'resources_per_trial': {
                'cpu': 4,
                'gpu': int(CUDA)
            },
            'num_samples': N_EXPERIMENTS,
            'checkpoint_at_end': True,
            'config': {
                'lr': LR, # tune.uniform(0.001, 0.1),
                'beta1': BETA1  # tune.uniform(0.1, 0.9),
            }
        })
