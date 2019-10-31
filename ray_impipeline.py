"""Data processing pipeline."""

import functools
import importlib
import logging
import numpy as np
import os
import random
import time

import ray
from ray import tune

from ray.tune import Trainable, grid_search
from ray.tune.schedulers import AsyncHyperBandScheduler

import geomstats
import torch
import torch.autograd
import torch.optim
import torch.utils.data

import datasets
import nn
import toylosses
import train_utils

import warnings
warnings.filterwarnings("ignore")

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

REDIS_ADDRESS = '127.0.0.1:6379'
N_EXPERIMENTS = 2

DATASET_NAME = 'connectomes'

HOME_DIR = '/scratch/users/nmiolane'
OUTPUT_DIR = os.path.join(HOME_DIR, 'ray_output_%s' % DATASET_NAME)
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train_vae')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

DEBUG = False

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

# Seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# NN nn_architecture
IMG_SHAPE = (1, 15, 15)
if DATASET_NAME == 'mnist':
    IMG_SHAPE = (1, 28, 28)

NN_TYPE = 'fc'
assert NN_TYPE in ['fc', 'conv', 'conv_plus']

RECONSTRUCTIONS = 'l2'
assert RECONSTRUCTIONS in ['riem', 'l2', 'l2_inner', 'bce']

SPD_FEATURE = 'log_vector'
assert SPD_FEATURE in [
    'matrix', 'vector',
    'log', 'log_vector',
    'log_frechet', 'log_frechet_vector',
    'point']
# Note: if point, then use RECONSTRUCTION riem

LATENT_DIM = 2
DATA_DIM = functools.reduce((lambda x, y: x * y), IMG_SHAPE)
if SPD_FEATURE in ['vector', 'log_vector', 'log_frechet_vector']:
    N = IMG_SHAPE[1]
    DATA_DIM = int(N * (N + 1) / 2)
INNER_DIM = 120

NN_ARCHITECTURE = {
    'img_shape': IMG_SHAPE,
    'data_dim': DATA_DIM,
    'latent_dim': LATENT_DIM,
    'n_layers': 4,
    'with_skip': True,
    'inner_dim': INNER_DIM,
    'logvar': 0.,
    'nn_type': NN_TYPE,
    'with_sigmoid': False,
    'spd_feature': SPD_FEATURE}

# MC samples
N_VEM_ELBO = 1
N_VEM_IWELBO = 99
N_VAE = 1  # N_VEM_ELBO + N_VEM_IWELBO
N_IWAE = N_VEM_ELBO + N_VEM_IWELBO

# for IWELBO to estimate the NLL
N_MC_NLL = 2000

# Train

FRAC_VAL = 0.05

BATCH_SIZE = {
    'mnist': 20,
    'connectomes': 4,
    'connectomes_simu': 8,
    'connectomes_schizophrenia': 16}
PRINT_INTERVAL = 64

N_EPOCHS = 64
CKPT_PERIOD = 20
LR = {
    'mnist': 1e-3,
    'connectomes': 1e-5,
    'connectomes_simu': 1e-5  # 1e-5,
    }

N_BATCH_PER_EPOCH = 1e10

if DEBUG:
    N_EPOCHS = 2
    BATCH_SIZE = {
        'mnist': 8,
        'connectomes': 2,
        'connectomes_schizophrenia': 2,
        'connectomes_simu': 2}
    N_VEM_ELBO = 1
    N_VEM_IWELBO = 399
    N_VAE = 1
    N_IWAE = 400
    N_MC_NLL = 50
    CKPT_PERIOD = 1
    N_BATCH_PER_EPOCH = 3

WEIGHTS_INIT = 'xavier'

REGULARIZATIONS = ('kullbackleibler',)
TRAIN_PARAMS = {
    'lr': LR[DATASET_NAME],
    'batch_size': BATCH_SIZE[DATASET_NAME],
    'beta1': 0.5,
    'beta2': 0.999,
    'weights_init': WEIGHTS_INIT,
    'reconstructions': RECONSTRUCTIONS,
    'regularizations': REGULARIZATIONS
    }


class Train(Trainable):

    def _setup(self, config):

        train_params = TRAIN_PARAMS
        train_params['lr'] = config.get('lr')
        train_params['batch_size'] = config.get('batch_size')

        nn_architecture = NN_ARCHITECTURE
        nn_architecture['latent_dim'] = config.get('latent_dim')
        nn_architecture['n_layers'] = config.get('n_layers')
        nn_architecture['inner_dim'] = config.get('inner_dim')
        nn_architecture['logvar'] = config.get('logvar')

        train_dataset, val_dataset = datasets.get_datasets(
            dataset_name=DATASET_NAME,
            frac_val=FRAC_VAL,
            batch_size=BATCH_SIZE[DATASET_NAME],
            img_shape=IMG_SHAPE)

        logging.info(
            'Train tensor: %s' % train_utils.get_logging_shape(train_dataset))
        logging.info(
            'Val tensor: %s' % train_utils.get_logging_shape(val_dataset))

        train_dataset = train_utils.spd_feature_from_matrix(
            train_dataset,
            spd_feature=nn_architecture['spd_feature'])
        val_dataset = train_utils.spd_feature_from_matrix(
            val_dataset,
            spd_feature=nn_architecture['spd_feature'])

        logging.info(
            'Train feature: %s' % train_utils.get_logging_shape(train_dataset))
        logging.info(
            'Val feature: %s' % train_utils.get_logging_shape(val_dataset))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_params['batch_size'], shuffle=True, **KWARGS)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_params['batch_size'], shuffle=True, **KWARGS)

        os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
        importlib.reload(geomstats.backend)

        m, o, s, t, v = train_utils.init_training(
            train_dir=self.logdir,
            nn_architecture=nn_architecture,
            train_params=train_params)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.modules = modules
        self.optimizers = optimizers
        self.start_epoch = start_epoch

        self.train_losses_all_epochs = train_losses_all_epochs
        self.val_losses_all_epochs = val_losses_all_epochs

        self.train_params = train_params
        self.nn_architecture = nn_architecture

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
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_time = 0

        n_data = len(self.train_loader.dataset)
        n_batches = len(self.train_loader)
        for batch_idx, batch_data in enumerate(self.train_loader):
            if batch_idx == 0:
                shape = batch_data.shape
                logging.info(
                    'Starting Train Epoch %d with %d batches, ' % (
                        epoch, n_batches)
                    + 'each of shape: '
                    '(' + ('%s, ' * len(shape) % shape)[:-2] + ')')

                logging.info('Architecture: ')
                logging.info(self.nn_architecture)
                logging.info('Training parameters:')
                logging.info(self.train_params)

            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue
            start = time.time()

            # TODO(nina): Uniformize batch_data across datasets
            if DATASET_NAME == 'connectomes':
                batch_data = batch_data.to(DEVICE).float()
            else:
                batch_data = batch_data[0].to(DEVICE).float()
            n_batch_data = len(batch_data)

            if batch_idx == 0:
                logging.info(
                    'Memory allocated in CUDA: %d Bytes.'
                    % torch.cuda.memory_allocated())

            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

            encoder = self.modules['encoder']
            decoder = self.modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                mu, logvar, n_samples=N_VAE).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            recon_np = batch_recon.detach().cpu().numpy()
            for i in range(len(recon_np)-1):
                assert not np.all(
                    recon_np[i] == recon_np[i+1]), recon_np[i] == recon_np[i+1]

            # --- VAE: Train wrt Neg ELBO --- #
            batch_data = batch_data.view(-1, DATA_DIM)
            batch_data_expanded = batch_data.expand(
                N_VAE, n_batch_data, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_VAE*n_batch_data, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon, batch_logvarx,
                reconstruction_type=RECONSTRUCTIONS)
            loss_reconstruction.backward(retain_graph=True)

            # TODO(nina): Check: No pb with N_VAE here?
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld
            loss_regularization.backward()

            self.optimizers['encoder'].step()
            self.optimizers['decoder'].step()
            # ------------------------------- #

            neg_elbo = loss_reconstruction + loss_regularization
            if neg_elbo != neg_elbo:
                print('elbo, then, recon, then regu')
                print(neg_elbo)
                print(loss_reconstruction)
                print(loss_regularization)
                raise ValueError('neg_elbo! ')

            end = time.time()
            total_time += end - start

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Neg ELBO: {:.6f}'
                    + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        neg_elbo,
                        loss_reconstruction, loss_regularization))

            start = time.time()
            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        end = time.time()
        total_time += end - start

        logging.info('====> Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_neg_elbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_elbo'] = average_neg_elbo
        train_losses['total_time'] = total_time

        self.train_losses_all_epochs.append(train_losses)

    def _train(self):
        self._train_iteration()
        return self._test()

    def _test(self):
        epoch = self._iteration
        for module in self.modules.values():
            module.eval()

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_loglikelihood = 0
        total_time = 0

        n_data = len(self.val_loader.dataset)
        n_batches = len(self.val_loader)
        for batch_idx, batch_data in enumerate(self.val_loader):
            if batch_idx == 0:
                shape = batch_data.shape
                logging.info(
                    'Starting Val Epoch %d with %d batches, ' % (
                        epoch, n_batches)
                    + 'each of shape: '
                    '(' + ('%s, ' * len(shape) % shape)[:-2] + ')')
                logging.info('NN_TYPE: %s.' % NN_TYPE)
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue

            start = time.time()
            if DATASET_NAME == 'connectomes':
                batch_data = batch_data.to(DEVICE).float()
            else:
                batch_data = batch_data[0].to(DEVICE).float()
            n_batch_data = len(batch_data)

            encoder = self.modules['encoder']
            decoder = self.modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                mu, logvar, n_samples=1).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            batch_data = batch_data.view(-1, DATA_DIM)
            batch_data_expanded = batch_data.expand(
                N_VAE, n_batch_data, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_VAE*n_batch_data, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon, batch_logvarx,
                reconstruction_type=RECONSTRUCTIONS)

            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization
            end = time.time()
            total_time += end - start

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            # TODO(nina): Release memory after neg_iwelbo computation
            neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL,
                reconstruction_type=RECONSTRUCTIONS)

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Val Epoch: {} [{}/{} ({:.0f}%)]\tBatch Neg ELBO: {:.6f}'
                    + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        neg_elbo,
                        loss_reconstruction, loss_regularization))

            start = time.time()
            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            total_neg_loglikelihood += n_batch_data * neg_loglikelihood.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_loglikelihood = total_neg_loglikelihood / n_data
        end = time.time()
        total_time += end - start

        logging.info('====> Val Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_neg_elbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = average_neg_loglikelihood
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['total_time'] = total_time

        self.val_losses_all_epochs.append(val_losses)
        return {'average_loss_reconstruction': average_loss_reconstruction}

    def _save(self, checkpoint_dir=None):
        epoch = self._iteration
        train_utils.save_checkpoint(
            epoch=epoch,
            modules=self.modules,
            optimizers=self.optimizers,
            dir_path=checkpoint_dir,
            train_losses_all_epochs=self.train_losses_all_epochs,
            val_losses_all_epochs=self.val_losses_all_epochs,
            nn_architecture=self.nn_architecture,
            train_params=self.train_params)
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

    sched = AsyncHyperBandScheduler(
        time_attr='training_iteration',
        metric='average_loss_reconstruction',
        mode='min')
    analysis = tune.run(
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
            'num_samples': 2,
            'checkpoint_at_end': True,
            'config': {
                'batch_size': grid_search([4]),
                'lr': grid_search([0.00001]),
                'latent_dim': grid_search([2, 5, 10, 20, 40, 80]),
                'n_layers': grid_search([4, 8]),
                'inner_dim': grid_search([256]),
                'logvar': grid_search([-10., -5.])
            }
        })
