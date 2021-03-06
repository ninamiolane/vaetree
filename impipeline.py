"""Data processing pipeline."""

import datetime as dt
import functools
import importlib
import logging
import luigi
import numpy as np
import os
import pickle
import random
import sys
import time

import geomstats
import torch
import torch.autograd
from torch.nn import functional as F
import torch.optim
import torch.utils.data

import datasets
import nn
import toylosses
import train_utils

import warnings
warnings.filterwarnings("ignore")

DATASET_NAME = 'connectomes'

HOME_DIR = '/scratch/users/nmiolane'
OUTPUT_DIR = sys.argv[1]
TRAIN_VAE_DIR = os.path.join(OUTPUT_DIR, 'train_vae')
TRAIN_IWAE_DIR = os.path.join(OUTPUT_DIR, 'train_iwae')
TRAIN_VEM_DIR = os.path.join(OUTPUT_DIR, 'train_vem')
TRAIN_VEGAN_DIR = os.path.join(OUTPUT_DIR, 'train_vegan')
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

LATENT_DIM = int(sys.argv[2])
DATA_DIM = functools.reduce((lambda x, y: x * y), IMG_SHAPE)
if SPD_FEATURE in ['vector', 'log_vector', 'log_frechet_vector']:
    N = IMG_SHAPE[1]
    DATA_DIM = int(N * (N + 1) / 2)

NN_ARCHITECTURE = {
    'img_shape': IMG_SHAPE,
    'data_dim': DATA_DIM,
    'latent_dim': LATENT_DIM,
    'inner_dim': 4096,
    'nn_type': NN_TYPE,
    'with_sigmoid': False,
    'spd_feature': SPD_FEATURE}

# MC samples
N_VEM_ELBO = 1
N_VEM_IWELBO = 99
N_VAE = 1  # N_VEM_ELBO + N_VEM_IWELBO
N_IWAE = N_VEM_ELBO + N_VEM_IWELBO

# for IWELBO to estimate the NLL
N_MC_NLL = 500

# Train

FRAC_VAL = 0.05

BATCH_SIZE = {
    'mnist': 20,
    'connectomes': 8,
    'connectomes_simu': 8,
    'connectomes_schizophrenia': 16}
PRINT_INTERVAL = 64
torch.backends.cudnn.benchmark = True

N_EPOCHS = 160
CKPT_PERIOD = 40
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
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}


class TrainVAE(luigi.Task):
    train_dir = os.path.join(TRAIN_VAE_DIR)
    train_losses_path = os.path.join(
        TRAIN_VAE_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(
        TRAIN_VAE_DIR, 'val_losses.pkl')

    def requires(self):
        pass

    def train_vae(self, epoch, train_loader, modules, optimizers):
        for module in modules.values():
            module.train()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_time = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx == 0:
                shape = batch_data.shape
                logging.info(
                    'Starting Train Epoch %d with %d batches, ' % (
                        epoch, n_batches)
                    + 'each of shape: '
                    '(' + ('%s, ' * len(shape) % shape)[:-2] + ')')

                logging.info('Architecture: ')
                logging.info(NN_ARCHITECTURE)
                logging.info('Training parameters:')
                logging.info(TRAIN_PARAMS)

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

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

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

            optimizers['encoder'].step()
            optimizers['decoder'].step()
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
        return train_losses

    def val_vae(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_loglikelihood = 0
        total_time = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        for batch_idx, batch_data in enumerate(val_loader):
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

            encoder = modules['encoder']
            decoder = modules['decoder']

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
        return val_losses

    def run(self):
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)
            os.chmod(self.train_dir, 0o777)

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
            spd_feature=NN_ARCHITECTURE['spd_feature'])
        val_dataset = train_utils.spd_feature_from_matrix(
            val_dataset,
            spd_feature=NN_ARCHITECTURE['spd_feature'])

        logging.info(
            'Train feature: %s' % train_utils.get_logging_shape(train_dataset))
        logging.info(
            'Val feature: %s' % train_utils.get_logging_shape(val_dataset))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TRAIN_PARAMS['batch_size'], shuffle=True, **KWARGS)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=TRAIN_PARAMS['batch_size'], shuffle=True, **KWARGS)

        os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
        importlib.reload(geomstats.backend)

        m, o, s, t, v = train_utils.init_training(
            train_dir=self.train_dir,
            nn_architecture=NN_ARCHITECTURE,
            train_params=TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        for epoch in range(start_epoch, N_EPOCHS):
            train_losses = self.train_vae(
                epoch, train_loader, modules, optimizers)
            val_losses = self.val_vae(
                epoch, val_loader, modules)

            train_losses_all_epochs.append(train_losses)
            val_losses_all_epochs.append(val_losses)

            if epoch % CKPT_PERIOD == 0:
                train_utils.save_checkpoint(
                    epoch=epoch, modules=modules, optimizers=optimizers,
                    dir_path=self.train_dir,
                    train_losses_all_epochs=train_losses_all_epochs,
                    val_losses_all_epochs=val_losses_all_epochs,
                    nn_architecture=NN_ARCHITECTURE,
                    train_params=TRAIN_PARAMS)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses_all_epochs, pkl)
        with open(self.output()['val_losses'].path, 'wb') as pkl:
            pickle.dump(val_losses_all_epochs, pkl)

    def output(self):
        return {
            'train_losses': luigi.LocalTarget(self.train_losses_path),
            'val_losses': luigi.LocalTarget(self.val_losses_path)}


class TrainIWAE(luigi.Task):
    train_dir = os.path.join(TRAIN_IWAE_DIR)
    train_losses_path = os.path.join(
        TRAIN_IWAE_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(
        TRAIN_IWAE_DIR, 'val_losses.pkl')

    def requires(self):
        pass

    def train_iwae(self, epoch, train_loader, modules, optimizers):
        for module in modules.values():
            module.train()
        total_neg_iwelbo = 0
        total_time = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue

            start = time.time()

            if DATASET_NAME == 'cryo':
                batch_data = np.vstack(
                    [torch.unsqueeze(b, 0) for b in batch_data])
                batch_data = torch.Tensor(batch_data).to(DEVICE)
            else:
                batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)
            end = time.time()
            total_time += end - start

            # --- IWAE: Train wrt IWAE --- #
            start = time.time()
            batch_data = batch_data.view(-1, DATA_DIM)
            neg_iwelbo = toylosses.neg_iwelbo(
                decoder,
                batch_data,
                mu, logvar,
                n_is_samples=N_IWAE, reconstruction_type=RECONSTRUCTIONS)

            # print('Anyway: neg_iwelbo pipeline = ', neg_iwelbo)
            if neg_iwelbo != neg_iwelbo or neg_iwelbo > 1e6:
                # print('mu = ', mu)
                # print('logvar = ', logvar)
                z = nn.sample_from_q(mu, logvar).to(DEVICE)
                # print('z = ', z)
                batch_recon, batch_logvarx = decoder(z)
                # print('batch_logvarx = ', batch_logvarx)
                # print('batch_recon = ', batch_recon)
                print('neg_iwelbo pipeline = ', neg_iwelbo)
                neg_iwelbo_bis = toylosses.neg_iwelbo(
                    decoder,
                    batch_data,
                    mu, logvar,
                    n_is_samples=N_IWAE,
                    reconstruction_type=RECONSTRUCTIONS)
                print('neg_iwelbo bis = ', neg_iwelbo_bis)
                neg_iwelbo_ter = toylosses.neg_iwelbo(
                    decoder,
                    batch_data,
                    mu, torch.zeros_like(mu),
                    n_is_samples=N_IWAE,
                    reconstruction_type=RECONSTRUCTIONS)

                print('neg_iwelbo_ter = ', neg_iwelbo_ter)
                # neg_iwelbo = neg_iwelbo_ter  # HACK
                # raise ValueError()
            neg_iwelbo.backward()

            optimizers['encoder'].step()
            optimizers['decoder'].step()
            end = time.time()
            total_time += end - start
            # ---------------------------- #

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]'
                    '\tBatch NEG IWELBO: {:.6f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        neg_iwelbo))

            start = time.time()
            total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_neg_iwelbo = total_neg_iwelbo / n_data
        end = time.time()
        total_time += end - start

        logging.info('====> Epoch: {} Average NEG IWELBO: {:.4f}'.format(
                epoch, average_neg_iwelbo))

        train_losses = {}
        train_losses['neg_iwelbo'] = average_neg_iwelbo
        train_losses['total_time'] = total_time
        return train_losses

    def val_iwae(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_iwelbo = 0
        total_neg_loglikelihood = 0
        total_time = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        for batch_idx, batch_data in enumerate(val_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue

            start = time.time()
            if DATASET_NAME == 'cryo':
                batch_data = np.vstack(
                    [torch.unsqueeze(b, 0) for b in batch_data])
                batch_data = torch.Tensor(batch_data).to(DEVICE)
            else:
                batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)
            end = time.time()
            total_time += end - start

            z = nn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            batch_data = batch_data.view(-1, DATA_DIM)
            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx,
                reconstruction_type=RECONSTRUCTIONS)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization

            # --- IWAE --- #
            start = time.time()
            batch_data = batch_data.view(-1, DATA_DIM)
            neg_iwelbo = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_VEM_IWELBO,
                reconstruction_type=RECONSTRUCTIONS)
            end = time.time()
            total_time += end - start
            # ------------ #

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL,
                reconstruction_type=RECONSTRUCTIONS)

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Val Epoch: {} [{}/{} ({:.0f}%)]'
                    '\tBatch NEG IWELBO: {:.6f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        neg_iwelbo))

            start = time.time()
            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
            total_neg_loglikelihood += n_batch_data * neg_loglikelihood.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        average_neg_loglikelihood = total_neg_loglikelihood / n_data
        end = time.time()
        total_time += end - start

        logging.info('====> Val Epoch: {} Average NEG IWELBO: {:.4f}'.format(
                epoch, average_neg_iwelbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = average_neg_loglikelihood
        val_losses['neg_iwelbo'] = average_neg_iwelbo
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['total_time'] = total_time
        return val_losses

    def run(self):
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)
            os.chmod(self.train_dir, 0o777)

        train_dataset, val_dataset = datasets.get_datasets(
            dataset_name=DATASET_NAME,
            frac_val=FRAC_VAL,
            batch_size=BATCH_SIZE[DATASET_NAME],
            img_shape=IMG_SHAPE)

        logging.info(
            'Train tensor: %s' % train_utils.get_logging_shape(train_dataset))
        logging.info(
            'Val tensor: %s' % train_utils.get_logging_shape(val_dataset))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TRAIN_PARAMS['batch_size'], shuffle=True, **KWARGS)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=TRAIN_PARAMS['batch_size'], shuffle=True, **KWARGS)

        m, o, s, t, v = train_utils.init_training(
            train_dir=self.train_dir,
            nn_architecture=NN_ARCHITECTURE,
            train_params=TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        for epoch in range(start_epoch, N_EPOCHS):
            train_losses = self.train_iwae(
                epoch, train_loader, modules, optimizers)
            val_losses = self.val_iwae(
                epoch, val_loader, modules)

            train_losses_all_epochs.append(train_losses)
            val_losses_all_epochs.append(val_losses)

            train_utils.save_checkpoint(
                epoch, modules, optimizers, self.train_dir,
                train_losses_all_epochs, val_losses_all_epochs)

        for module_name, module in modules.items():
            module_path = os.path.join(
                self.train_dir,
                '{}.pth'.format(module_name))
            torch.save(module, module_path)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses_all_epochs, pkl)
        with open(self.output()['val_losses'].path, 'wb') as pkl:
            pickle.dump(val_losses_all_epochs, pkl)

    def output(self):
        return {
            'train_losses': luigi.LocalTarget(self.train_losses_path),
            'val_losses': luigi.LocalTarget(self.val_losses_path)}


class TrainVEM(luigi.Task):
    train_dir = os.path.join(TRAIN_VEM_DIR)
    train_losses_path = os.path.join(TRAIN_VEM_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(TRAIN_VEM_DIR, 'val_losses.pkl')

    def requires(self):
        pass

    def train_vem(self, epoch, train_loader, modules, optimizers):
        for module in modules.values():
            module.train()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_iwelbo = 0
        total_time = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue

            start = time.time()
            if DATASET_NAME == 'cryo':
                batch_data = np.vstack(
                    [torch.unsqueeze(b, 0) for b in batch_data])
                batch_data = torch.Tensor(batch_data).to(DEVICE)
            else:
                batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)
            batch_data = batch_data.view(n_batch_data, DATA_DIM)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                mu, logvar, n_samples=N_VEM_ELBO).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            # --- VEM: Train encoder with NEG ELBO, proxy for KL --- #
            batch_data = batch_data.view(-1, DATA_DIM)
            half = int(n_batch_data / 2)
            batch_data_first_half = batch_data[:half, ]
            batch_recon_first_half = batch_recon[:half, ]
            batch_logvarx_first_half = batch_logvarx[:half, ]

            batch_data_expanded = batch_data_first_half.expand(
                N_VEM_ELBO, half, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_VEM_ELBO*half, DATA_DIM)
            batch_recon_expanded = batch_recon_first_half.expand(
                N_VEM_ELBO, half, DATA_DIM)
            batch_recon_flat = batch_recon_expanded.resize(
                N_VEM_ELBO*half, DATA_DIM)
            batch_logvarx_expanded = batch_logvarx_first_half.expand(
                N_VEM_ELBO, half, DATA_DIM)
            batch_logvarx_flat = batch_logvarx_expanded.resize(
                N_VEM_ELBO*half, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon_flat, batch_logvarx_flat,
                reconstruction_type=RECONSTRUCTIONS)
            if loss_reconstruction is None or loss_reconstruction > 5e4:
                print('Error in loss recon', loss_reconstruction)
                batch_recon, batch_logvarx = decoder(mu)
                batch_data = batch_data.view(-1, DATA_DIM)
                half = int(n_batch_data / 2)
                batch_data_first_half = batch_data[:half, ]
                batch_recon_first_half = batch_recon[:half, ]
                batch_logvarx_first_half = batch_logvarx[:half, ]

                batch_data_expanded = batch_data_first_half.expand(
                    N_VEM_ELBO, half, DATA_DIM)
                batch_data_flat = batch_data_expanded.resize(
                    N_VEM_ELBO*half, DATA_DIM)
                batch_recon_expanded = batch_recon_first_half.expand(
                    N_VEM_ELBO, half, DATA_DIM)
                batch_recon_flat = batch_recon_expanded.resize(
                    N_VEM_ELBO*half, DATA_DIM)
                batch_logvarx_expanded = batch_logvarx_first_half.expand(
                    N_VEM_ELBO, half, DATA_DIM)
                batch_logvarx_flat = batch_logvarx_expanded.resize(
                    N_VEM_ELBO*half, DATA_DIM)

                loss_reconstruction = toylosses.reconstruction_loss(
                    batch_data_flat, batch_recon_flat, batch_logvarx_flat,
                    reconstruction_type=RECONSTRUCTIONS)
            loss_reconstruction.backward(retain_graph=True)

            loss_regularization = toylosses.regularization_loss(
                mu[:half, ], logvar[:half, ])  # kld
            loss_regularization.backward(retain_graph=True)

            optimizers['encoder'].step()
            neg_elbo = loss_reconstruction + loss_regularization

            # ----------------------------------------------------- #

            # --- VEM: Train decoder with IWAE, proxy for NLL --- #
            half = 0   # DEBUG - delete this line
            batch_data_second_half = batch_data[half:, ]
            batch_data_second_half = batch_data_second_half.view(-1, DATA_DIM)

            optimizers['decoder'].zero_grad()

            neg_iwelbo = toylosses.neg_iwelbo(
                decoder,
                batch_data_second_half,
                mu[half:, ], logvar[half:, ],
                n_is_samples=N_VEM_IWELBO,
                reconstruction_type=RECONSTRUCTIONS)
            # This also fills the encoder, but we do not step
            neg_iwelbo.backward()
            if neg_iwelbo != neg_iwelbo or neg_iwelbo > 5e4:
                print('mu = ', mu)
                print('logvar = ', logvar)
                z = nn.sample_from_q(mu, logvar).to(DEVICE)
                print('z = ', z)
                batch_recon, batch_logvarx = decoder(z)
                print('batch_logvarx = ', batch_logvarx)
                print('batch_recon = ', batch_recon)
                print('neg_iwelbo pipeline = ', neg_iwelbo)
                neg_iwelbo_bis = toylosses.neg_iwelbo(
                    decoder,
                    batch_data,
                    mu, logvar,
                    n_is_samples=N_IWAE,
                    reconstruction_type=RECONSTRUCTIONS)
                print('neg_iwelbo bis = ', neg_iwelbo_bis)
                neg_iwelbo_ter = toylosses.neg_iwelbo(
                    decoder,
                    batch_data,
                    mu, torch.zeros_like(mu),
                    n_is_samples=N_IWAE,
                    reconstruction_type=RECONSTRUCTIONS)

                neg_iwelbo = neg_iwelbo_ter  # HACK
            optimizers['decoder'].step()
            # ----------------------------------------------------- #
            end = time.time()
            total_time += end - start

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            # neg_loglikelihood = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL,
            #     reconstruction_type=RECONSTRUCTIONS)

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]'
                    '\tBatch Neg ELBO: {:.6f}'
                    '\tBatch Neg IWELBO: {:.4f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        neg_elbo,
                        neg_iwelbo))

            start = time.time()
            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
            # total_neg_loglikelihood +=n_batch_data * neg_loglikelihood.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        # average_neg_loglikelihood = total_neg_loglikelihood / n_data
        end = time.time()
        total_time += end - start

        logging.info(
            '====> Epoch: {} '
            'Average Neg ELBO: {:.4f}'
            'Average Neg IWELBO: {:.4f}'.format(
                epoch, average_neg_elbo, average_neg_iwelbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = 0.  # average_neg_loglikelihood
        train_losses['neg_elbo'] = average_neg_elbo
        train_losses['neg_iwelbo'] = average_neg_iwelbo
        train_losses['total_time'] = total_time

        return train_losses

    def val_vem(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_iwelbo = 0
        total_neg_loglikelihood = 0
        total_time = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        for batch_idx, batch_data in enumerate(val_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue

            start = time.time()
            if DATASET_NAME == 'cryo':
                batch_data = np.vstack(
                    [torch.unsqueeze(b, 0) for b in batch_data])
                batch_data = torch.Tensor(batch_data).to(DEVICE)
            else:
                batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)
            batch_data = batch_data.view(n_batch_data, DATA_DIM)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                mu, logvar, n_samples=N_VEM_ELBO).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            # --- VEM: Neg ELBO
            half = int(n_batch_data / 2)
            batch_data = batch_data.view(-1, DATA_DIM)
            batch_data_first_half = batch_data[:half, ]
            batch_recon_first_half = batch_recon[:half, ]
            batch_logvarx_first_half = batch_logvarx[:half, ]

            batch_data_expanded = batch_data_first_half.expand(
                N_VEM_ELBO, half, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_VEM_ELBO*half, DATA_DIM)
            batch_recon_expanded = batch_recon_first_half.expand(
                N_VEM_ELBO, half, DATA_DIM)
            batch_recon_flat = batch_recon_expanded.resize(
                N_VEM_ELBO*half, DATA_DIM)
            batch_logvarx_expanded = batch_logvarx_first_half.expand(
                N_VEM_ELBO, half, DATA_DIM)
            batch_logvarx_flat = batch_logvarx_expanded.resize(
                N_VEM_ELBO*half, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon_flat, batch_logvarx_flat,
                reconstruction_type=RECONSTRUCTIONS)

            loss_reconstruction.backward(retain_graph=True)
            loss_regularization = toylosses.regularization_loss(
                mu[:half, ], logvar[:half, ])  # kld
            neg_elbo = loss_reconstruction + loss_regularization

            # --- VEM: Neg IWELBO
            batch_data_second_half = batch_data[half:, ]
            neg_iwelbo = toylosses.neg_iwelbo(
                decoder,
                batch_data_second_half,
                mu[half:, ], logvar[half:, ],
                n_is_samples=N_VEM_IWELBO,
                reconstruction_type=RECONSTRUCTIONS)
            end = time.time()
            total_time += end - start

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL,
                reconstruction_type=RECONSTRUCTIONS)

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Val Epoch: {} [{}/{} ({:.0f}%)]'
                    '\tBatch Neg ELBO: {:.6f}'
                    '\tBatch Neg IWELBO: {:.4f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        neg_elbo,
                        neg_iwelbo))

            start = time.time()
            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
            total_neg_loglikelihood += n_batch_data * neg_loglikelihood.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        average_neg_loglikelihood = total_neg_loglikelihood / n_data
        end = time.time()
        total_time += end - start

        logging.info(
            '====> Val Epoch: {} '
            'Average Negative ELBO: {:.4f}'
            'Average IWAE: {:.4f}'.format(
                epoch, average_neg_elbo, average_neg_iwelbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = average_neg_loglikelihood
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['neg_iwelbo'] = average_neg_iwelbo
        val_losses['total_time'] = total_time
        return val_losses

    def run(self):
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)
            os.chmod(self.train_dir, 0o777)

        train_dataset, val_dataset = datasets.get_datasets(
            dataset_name=DATASET_NAME,
            frac_val=FRAC_VAL,
            batch_size=BATCH_SIZE[DATASET_NAME],
            img_shape=IMG_SHAPE)

        logging.info(
            'Train tensor: %s' % train_utils.get_logging_shape(train_dataset))
        logging.info(
            'Val tensor: %s' % train_utils.get_logging_shape(val_dataset))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TRAIN_PARAMS['batch_size'], shuffle=True, **KWARGS)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=TRAIN_PARAMS['batch_size'], shuffle=True, **KWARGS)

        m, o, s, t, v = train_utils.init_training(
            train_dir=self.train_dir,
            nn_architecture=NN_ARCHITECTURE,
            train_params=TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        for epoch in range(start_epoch, N_EPOCHS):
            train_losses = self.train_vem(
                epoch, train_loader, modules, optimizers)
            train_losses_all_epochs.append(train_losses)
            val_losses = self.val_vem(
                epoch, val_loader, modules)
            val_losses_all_epochs.append(val_losses)

            train_utils.save_checkpoint(
                epoch, modules, optimizers, self.train_dir,
                train_losses_all_epochs, val_losses_all_epochs)

        for module_name, module in modules.items():
            module_path = os.path.join(
                self.train_dir,
                '{}.pth'.format(module_name))
            torch.save(module, module_path)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses_all_epochs, pkl)
        with open(self.output()['val_losses'].path, 'wb') as pkl:
            pickle.dump(val_losses_all_epochs, pkl)

    def output(self):
        return {
            'train_losses': luigi.LocalTarget(self.train_losses_path),
            'val_losses': luigi.LocalTarget(self.val_losses_path)}


class TrainVEGAN(luigi.Task):
    train_dir = os.path.join(TRAIN_VEGAN_DIR, 'models')
    train_losses_path = os.path.join(TRAIN_VEGAN_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(TRAIN_VEGAN_DIR, 'val_losses.pkl')

    def requires(self):
        pass

    def train_vegan(self, epoch, train_loader, modules, optimizers):
        for module in modules.values():
            module.train()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_loss_discriminator = 0
        total_loss_generator = 0
        total_loss = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)

        for batch_idx, batch_data in enumerate(train_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue

            batch_data = batch_data[0].to(DEVICE)
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                    mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = nn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, batch_logvarx_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx,
                reconstruction_type=RECONSTRUCTIONS)
            loss_reconstruction.backward(retain_graph=True)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld
            loss_regularization.backward()

            # - ENCODER STEP -
            optimizers['encoder'].step()
            # - - - - - - - - -

            n_from_prior = int(np.random.uniform(
                low=n_batch_data - n_batch_data / 4,
                high=n_batch_data + n_batch_data / 4))
            z_from_prior = nn.sample_from_prior(
                LATENT_DIM, n_samples=n_from_prior)
            batch_recon_from_prior, batch_logvarx_from_prior = decoder(
                z_from_prior)

            discriminator = modules['discriminator']

            real_labels = torch.full((n_batch_data, 1), 1, device=DEVICE)
            fake_labels = torch.full((n_batch_data, 1), 0, device=DEVICE)

            # -- Update Discriminator
            labels_data = discriminator(batch_data)
            labels_from_prior = discriminator(batch_from_prior)

            loss_dis_data = F.binary_cross_entropy(
                        labels_data,
                        real_labels)
            loss_dis_from_prior = F.binary_cross_entropy(
                        labels_from_prior,
                        fake_labels)

            loss_discriminator = (
                    loss_dis_data + loss_dis_from_prior)

            # Fill gradients on discriminator only
            loss_discriminator.backward(retain_graph=True)

            # - DISCRIMINATOR STEP -
            # Before filing with gradient of loss_generator
            optimizers['discriminator'].step()
            # - - - - - - - - -

            # -- Update Generator/Decoder
            loss_generator = F.binary_cross_entropy(
                    labels_from_prior,
                    real_labels)

            # Fill gradients on generator only
            # FREE THE DECODER:
            optimizers['decoder'].zero_grad()
            # Only back propagate the loss of the generator through the deocder
            loss_generator.backward()

            # - DECODER STEP -
            optimizers['decoder'].step()
            # - - - - - - - - -

            loss = loss_reconstruction + loss_regularization
            # loss += loss_discriminator + loss_generator

            if batch_idx % PRINT_INTERVAL == 0:
                dx = labels_data.mean()
                dgz = labels_from_prior.mean()

                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'
                    + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
                string_base += (
                    ', Discriminator: {:.6f}; Generator: {:.6f},'
                    '\nD(x): {:.3f}, D(G(z)): {:.3f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        loss,
                        loss_reconstruction,
                        loss_regularization,
                        loss_discriminator,
                        loss_generator,
                        dx, dgz))

            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_loss_discriminator += (
                n_batch_data * loss_discriminator.item())
            total_loss_generator += (n_batch_data * loss_generator.item())

            total_loss += n_batch_data * loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_loss = total_loss / n_data

        # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
        neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL,
                reconstruction_type=RECONSTRUCTIONS)

        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, average_loss))
        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = neg_loglikelihood
        train_losses['total'] = average_loss
        return train_losses

    def val_vegan(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_loss_discriminator = 0
        total_loss_generator = 0
        total_loss = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)

        for batch_idx, batch_data in enumerate(val_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue

            batch_data = batch_data[0].to(DEVICE)
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                    mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = nn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, batch_logvarx_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx,
                reconstruction_type=RECONSTRUCTIONS)
            loss_reconstruction.backward(retain_graph=True)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            n_from_prior = int(np.random.uniform(
                low=n_batch_data - n_batch_data / 4,
                high=n_batch_data + n_batch_data / 4))
            z_from_prior = nn.sample_from_prior(
                LATENT_DIM, n_samples=n_from_prior)
            batch_recon_from_prior, batch_logvarx_from_prior = decoder(
                z_from_prior)

            discriminator = modules['discriminator']

            real_labels = torch.full((n_batch_data, 1), 1, device=DEVICE)
            fake_labels = torch.full((n_batch_data, 1), 0, device=DEVICE)

            # -- Update Discriminator
            labels_data = discriminator(batch_data)
            labels_from_prior = discriminator(batch_from_prior)

            loss_dis_data = F.binary_cross_entropy(
                        labels_data,
                        real_labels)
            loss_dis_from_prior = F.binary_cross_entropy(
                        labels_from_prior,
                        fake_labels)

            loss_discriminator = (
                    loss_dis_data + loss_dis_from_prior)

            # -- Update Generator/Decoder
            loss_generator = F.binary_cross_entropy(
                    labels_from_prior,
                    real_labels)

            loss = loss_reconstruction + loss_regularization
            # loss += loss_discriminator + loss_generator

            if batch_idx % PRINT_INTERVAL == 0:
                dx = labels_data.mean()
                dgz = labels_from_prior.mean()

                string_base = (
                    'Val Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'
                    + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
                string_base += (
                    ', Discriminator: {:.6f}; Generator: {:.6f},'
                    '\nD(x): {:.3f}, D(G(z)): {:.3f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        loss,
                        loss_reconstruction,
                        loss_regularization,
                        loss_discriminator,
                        loss_generator,
                        dx, dgz))

            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_loss_discriminator += (
                n_batch_data * loss_discriminator.item())
            total_loss_generator += n_batch_data * loss_generator.item()

            total_loss += n_batch_data * loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_loss = total_loss / n_data

        # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
        neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL,
                reconstruction_type=RECONSTRUCTIONS)

        logging.info('====> Val Epoch: {} Average loss: {:.4f}'.format(
                epoch, average_loss))
        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = neg_loglikelihood
        val_losses['total'] = average_loss
        return val_losses

    def run(self):
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)
            os.chmod(self.train_dir, 0o777)

        train_dataset, val_dataset = datasets.get_datasets(
            dataset_name=DATASET_NAME,
            frac_val=FRAC_VAL,
            batch_size=BATCH_SIZE[DATASET_NAME],
            img_shape=IMG_SHAPE)

        logging.info(
            'Train tensor: %s' % train_utils.get_logging_shape(train_dataset))
        logging.info(
            'Val tensor: %s' % train_utils.get_logging_shape(val_dataset))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TRAIN_PARAMS['batch_size'], shuffle=True, **KWARGS)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=TRAIN_PARAMS['batch_size'], shuffle=True, **KWARGS)

        m, o, s, t, v = train_utils.init_training(
            train_dir=self.train_dir,
            nn_architecture=NN_ARCHITECTURE,
            train_params=TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        for epoch in range(start_epoch, N_EPOCHS):
            train_losses = self.train_vegan(
                epoch, train_loader, modules, optimizers)
            train_losses_all_epochs.append(train_losses)
            val_losses = self.val_vegan(
                epoch, val_loader, modules)
            val_losses_all_epochs.append(val_losses)

        for module_name, module in modules.items():
            module_path = os.path.join(
                self.train_dir,
                '{}.pth'.format(module_name))
            torch.save(module, module_path)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses_all_epochs, pkl)
        with open(self.output()['val_losses'].path, 'wb') as pkl:
            pickle.dump(val_losses_all_epochs, pkl)

    def output(self):
        return {
            'train_losses': luigi.LocalTarget(self.train_losses_path),
            'val_losses': luigi.LocalTarget(self.val_losses_path)}


class RunAll(luigi.Task):
    def requires(self):
        return TrainVAE()  # , TrainIWAE(), TrainVEM()

    def output(self):
        return luigi.LocalTarget('dummy')


def init():
    directories = [
        OUTPUT_DIR,
        TRAIN_VAE_DIR, TRAIN_IWAE_DIR, TRAIN_VEM_DIR,
        REPORT_DIR]
    for directory in directories:
        if not os.path.isdir(directory):
            os.mkdir(directory)
            os.chmod(directory, 0o777)

    logs_filename = os.path.join(OUTPUT_DIR, 'logs%s.txt' % dt.datetime.now())
    logging.basicConfig(
        filename=logs_filename,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        level=logging.INFO)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logging.info('start')
    luigi.run(
        main_task_cls=RunAll(),
        cmdline_args=[
            '--local-scheduler',
        ])


if __name__ == "__main__":
    init()
