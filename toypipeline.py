"""Data processing pipeline."""

import datetime as dt
import importlib
import logging
import luigi
import numpy as np
import os
import pickle
import random
import time
import sys

import ray
from ray import tune

from ray.tune import Trainable, grid_search
from ray.tune.schedulers import AsyncHyperBandScheduler

import geomstats
import torch
import torch.autograd
from torch.nn import functional as F
import torch.optim
import torch.utils.data

import datasets
import toylosses
import toynn
import train_utils

import warnings
warnings.filterwarnings("ignore")

DATASET_NAME = 'synthetic'

HOME_DIR = '/scratch/users/nmiolane'
OUTPUT_DIR = sys.argv[1]

SYNTHETIC_DIR = os.path.join(OUTPUT_DIR, 'synthetic')
TRAIN_VAE_DIR = os.path.join(OUTPUT_DIR, 'train_vae')
TRAIN_IWAE_DIR = os.path.join(OUTPUT_DIR, 'train_iwae')
TRAIN_VEM_DIR = os.path.join(OUTPUT_DIR, 'train_vem')
TRAIN_VEGAN_DIR = os.path.join(OUTPUT_DIR, 'train_vegan')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

DEBUG = False

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# NN architecture

W_TRUE, B_TRUE, NONLINEARITY = (
 {0: [[3.0], [-2.0]],
  1: [[.05, -.05], [-.15, -.1]]},
 {0: [1.0, 3.0],
  1: [0.062, 0.609]},
 'softplus')

DATA_DIM = 2
LATENT_DIM = 1
LOGVARX_TRUE = float(sys.argv[2])
N_SAMPLES = int(sys.argv[3])  # Number of synthetic data
MANIFOLD_NAME = sys.argv[4]
VAE_TYPE = 'gvae'

if DEBUG:
    N_SAMPLES = 20

NN_ARCHITECTURE = {
    'nn_type': 'toy',
    'data_dim': DATA_DIM,
    'latent_dim': LATENT_DIM,
    'n_decoder_layers': 2,
    'nonlinearity': NONLINEARITY,
    'with_biasx': True,
    'with_logvarx': False,
    'with_biasz': True,
    'with_logvarz': True,
    'logvarx_true': LOGVARX_TRUE,
    'manifold_name': MANIFOLD_NAME}

# MC samples for AVEM
N_MC_ELBO = 1
N_MC_IWELBO = 9
N_MC_TOT = 1  # N_MC_ELBO + N_MC_IWELBO
BATCH_SIZE = 256
if DEBUG:
    BATCH_SIZE = 4

TRAIN_PARAMS = {
    'weights_init': 'xavier',
    'lr': 5e-3,
    'batch_size': BATCH_SIZE,
    'beta1': 0.5,
    'beta2': 0.999,
    'n_mc_tot': N_MC_TOT,
    'logvarx_true': LOGVARX_TRUE,
    'reconstruction_type': 'riem'}


if NN_ARCHITECTURE['with_logvarx']:
    assert len(W_TRUE) == NN_ARCHITECTURE['n_decoder_layers'] + 1, len(W_TRUE)
else:
    assert len(W_TRUE) == NN_ARCHITECTURE['n_decoder_layers']

# Train
FRAC_VAL = 0.2

PRINT_PERIOD = 16
CKPT_PERIOD = 1
N_EPOCHS = 101


class Train(Trainable):
    train_dir = TRAIN_VAE_DIR
    models_path = os.path.join(TRAIN_VAE_DIR, 'models')
    train_losses_path = os.path.join(
        TRAIN_VAE_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(
        TRAIN_VAE_DIR, 'val_losses.pkl')

    def _setup(self, config):

        train_params = TRAIN_PARAMS
        train_params['lr'] = config.get('lr')
        train_params['batch_size'] = config.get('batch_size')

        nn_architecture = NN_ARCHITECTURE
        nn_architecture['latent_dim'] = config.get('latent_dim')

        train_dataset, val_dataset = datasets.get_datasets(
            dataset_name=DATASET_NAME,
            nn_architecture=nn_architecture)

        logging.info(
            'Train tensor: %s' % train_utils.get_logging_shape(train_dataset))
        logging.info(
            'Val tensor: %s' % train_utils.get_logging_shape(val_dataset))

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

    def _train_iteration(self, epoch, train_loader, modules, optimizers):
        """
        A train iteration for algo_name == 'vae' here.
        """
        epoch = self._iteration
        algo_name = self.train_params['algo_name']

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
            if DEBUG and batch_idx > 3:
                continue
            start = time.time()

            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']
            #print('vefore encod')
            mu, logvar = encoder(batch_data)

            z = toynn.sample_from_q(
                mu, logvar, n_samples=N_MC_TOT).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)
            #print('before expand')
            # --- VAE: Train wrt Neg ELBO --- #
            batch_data_expanded = batch_data.expand(
                N_MC_TOT, n_batch_data, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_MC_TOT*n_batch_data, DATA_DIM)
            #print('before rec loss')  # this one we see
            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat,
                batch_recon,
                batch_logvarx,
                reconstruction_type=TRAIN_PARAMS['reconstruction_type'],
                manifold_name=MANIFOLD_NAME)
            loss_reconstruction.backward(retain_graph=True)

            #print('before reg loss')
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld
            loss_regularization.backward()
            #print('before opt')

            optimizers['encoder'].step()
            optimizers['decoder'].step()
            #print('end compt')
            # ------------------------------- #

            neg_elbo = loss_reconstruction + loss_regularization

            end = time.time()
            total_time += end - start

            neg_iwelbo = torch.Tensor([0.]) #toylosses.neg_iwelbo(
                #decoder, batch_data, mu, logvar, n_is_samples=5000) #   N_MC_TOT)

            if batch_idx % PRINT_PERIOD == 0:
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
            total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        end = time.time()
        total_time += end - start

        weight_w = decoder.layers[0].weight[[0]]
        weight_phi = encoder.fc1.weight[[0]]
        # train_data = torch.Tensor(train_loader.dataset)
        # neg_loglikelihood = toylosses.fa_neg_loglikelihood(
        #    weight_w, train_data)

        logging.info('====> Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_neg_elbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = 0  # neg_loglikelihood
        train_losses['neg_elbo'] = average_neg_elbo
        train_losses['neg_iwelbo'] = average_neg_iwelbo
        train_losses['weight_w'] = weight_w
        train_losses['weight_phi'] = weight_phi
        train_losses['total_time'] = total_time

        self.train_losses_all_epochs.append(train_losses)

    def _train(self):
        self._train_iteration()
        return self._test()

    def _test(self, epoch, val_loader, modules, algo_name='vae'):
        epoch = self._iteration
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_iwelbo = 0
        total_time = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        for batch_idx, batch_data in enumerate(val_loader):
            if DEBUG:
                if batch_idx > 3:
                    continue

            start = time.time()
            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = toynn.sample_from_q(
                mu, logvar, n_samples=N_MC_TOT).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            batch_data_expanded = batch_data.expand(
                N_MC_TOT, n_batch_data, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_MC_TOT*n_batch_data, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat,
                batch_recon,
                batch_logvarx,
                reconstruction_type=TRAIN_PARAMS['reconstruction_type'],
                manifold_name=MANIFOLD_NAME)

            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization
            end = time.time()
            total_time += end - start

            neg_iwelbo = torch.Tensor([0.]) #toylosses.neg_iwelbo(
                #decoder, batch_data, mu, logvar, n_is_samples=5000)  #N_MC_TOT)

            if batch_idx % PRINT_PERIOD == 0:
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
            total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        end = time.time()
        total_time += end - start

        weight = decoder.layers[0].weight[[0]]
        # val_data = torch.Tensor(val_loader.dataset)
        # neg_loglikelihood = toylosses.fa_neg_loglikelihood(weight, val_data)

        logging.info('====> Val Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_neg_elbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = 0  # neg_loglikelihood
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['neg_iwelbo'] = average_neg_iwelbo
        val_losses['weight'] = weight
        val_losses['total_time'] = total_time

        self.val_losses_all_epochs.append(val_losses)
        return val_losses

    def _save(self, checkpoint_dir=None):
        epoch = self._iteration

        train_utils.save_checkpoint(
            epoch=epoch,
            modules=self.modules,
            optimizers=self.optimizers,
            dir_path=self.train_dir,
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
                algo_name=self.algo_name,
                module_name=module_name,
                epoch_id=epoch_id)


class TrainIWAE(luigi.Task):
    train_dir = TRAIN_IWAE_DIR
    models_path = os.path.join(TRAIN_IWAE_DIR, 'models')
    train_losses_path = os.path.join(
        TRAIN_IWAE_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(
        TRAIN_IWAE_DIR, 'val_losses.pkl')

    def requires(self):
        return MakeDataSet()

    def train_iwae(self, epoch, train_loader, modules, optimizers):
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
            if DEBUG:
                if batch_idx > 3:
                    continue

            start = time.time()

            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)
            end = time.time()
            total_time += end - start

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization

            # --- IWAE: Train wrt IWAE --- #
            start = time.time()
            neg_iwelbo = toylosses.neg_iwelbo(
                decoder,
                batch_data,
                mu, logvar,
                n_is_samples=N_MC_TOT)

            neg_iwelbo.backward()

            optimizers['encoder'].step()
            optimizers['decoder'].step()
            end = time.time()
            total_time += end - start
            # ---------------------------- #

            if batch_idx % PRINT_PERIOD == 0:
                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]'
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
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        end = time.time()
        total_time += end - start

        weight_w = decoder.layers[0].weight[[0]]
        weight_phi = encoder.fc1.weight[[0]]
        #train_data = torch.Tensor(train_loader.dataset)
        #neg_loglikelihood = toylosses.fa_neg_loglikelihood(
        #    weight_w, train_data)

        logging.info('====> Epoch: {} Average NEG IWELBO: {:.4f}'.format(
                epoch, average_neg_iwelbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = 0 #neg_loglikelihood
        train_losses['neg_iwelbo'] = average_neg_iwelbo
        train_losses['neg_elbo'] = average_neg_elbo
        train_losses['weight_w'] = weight_w
        train_losses['weight_phi'] = weight_phi
        train_losses['total_time'] = total_time
        return train_losses

    def val_iwae(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_iwelbo = 0
        total_time = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        for batch_idx, batch_data in enumerate(val_loader):
            if DEBUG:
                if batch_idx > 3:
                    continue

            start = time.time()
            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)
            end = time.time()
            total_time += end - start

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization

            # --- IWAE --- #
            start = time.time()
            neg_iwelbo = toylosses.neg_iwelbo(
                decoder,
                batch_data,
                mu, logvar,
                n_is_samples=N_MC_TOT)
            end = time.time()
            total_time += end - start
            # ------------ #

            if batch_idx % PRINT_PERIOD == 0:
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
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        end = time.time()
        total_time += end - start

        weight = decoder.layers[0].weight[[0]]
        #val_data = torch.Tensor(val_loader.dataset)
        #neg_loglikelihood = toylosses.fa_neg_loglikelihood(weight, val_data)

        logging.info('====> Val Epoch: {} Average NEG IWELBO: {:.4f}'.format(
                epoch, average_neg_iwelbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = 0 #neg_loglikelihood
        val_losses['neg_iwelbo'] = average_neg_iwelbo
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['weight'] = weight
        val_losses['total_time'] = total_time
        return val_losses

    def run(self):
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        dataset_path = self.input().path
        dataset = torch.Tensor(np.load(dataset_path))

        logging.info('--Dataset tensor: (%d, %d)' % dataset.shape)

        n_train = int((1 - FRAC_VAL) * N_SAMPLES)
        train = torch.Tensor(dataset[:n_train, :])
        val = torch.Tensor(dataset[n_train:, :])

        logging.info('-- Train tensor: (%d, %d)' % train.shape)
        logging.info('-- Validation tensor: (%d, %d)' % val.shape)

        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)
        val_dataset = torch.utils.data.TensorDataset(val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        m, o, s, t, v = train_utils.init_training(
            self.train_dir, NN_ARCHITECTURE, TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v
        for epoch in range(start_epoch, N_EPOCHS):
            if DEBUG:
                if epoch > 2:
                    break
            train_losses = self.train_iwae(
                epoch, train_loader, modules, optimizers)
            val_losses = self.val_iwae(
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

        for module_name, module in modules.items():
            module_path = os.path.join(
                self.models_path,
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
    train_dir = TRAIN_VEM_DIR
    models_path = os.path.join(TRAIN_VEM_DIR, 'models')
    train_losses_path = os.path.join(TRAIN_VEM_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(TRAIN_VEM_DIR, 'val_losses.pkl')

    def requires(self):
        return MakeDataSet()

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
            if DEBUG:
                if batch_idx > 3:
                    continue

            start = time.time()
            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = toynn.sample_from_q(
                mu, logvar, n_samples=N_MC_ELBO).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            # --- VEM: Train encoder with NEG ELBO, proxy for KL --- #
            half = int(n_batch_data / 2)
            batch_data_first_half = batch_data[:half, ]

            batch_data_expanded = batch_data_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)

            batch_recon_first_half = batch_recon[:half, ]
            batch_recon_expanded = batch_recon_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_recon_flat = batch_recon_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)

            batch_logvarx_first_half = batch_logvarx[:half, ]
            batch_logvarx_expanded = batch_logvarx_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_logvarx_flat = batch_logvarx_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon_flat, batch_logvarx_flat)

            loss_reconstruction.backward(retain_graph=True)

            loss_regularization = toylosses.regularization_loss(
                mu[:half, ], logvar[:half, ])  # kld
            loss_regularization.backward(retain_graph=True)

            optimizers['encoder'].step()
            neg_elbo = loss_reconstruction + loss_regularization
            # ----------------------------------------------------- #

            # --- VEM: Train decoder with IWAE, proxy for NLL --- #
            batch_data_second_half = batch_data[half:, ]

            optimizers['decoder'].zero_grad()

            neg_iwelbo = toylosses.neg_iwelbo(
                decoder,
                batch_data_second_half,
                mu[half:, ], logvar[half:, ],
                n_is_samples=N_MC_IWELBO)
            # This also fills the encoder, but we do not step
            neg_iwelbo.backward()
            optimizers['decoder'].step()
            # ----------------------------------------------------- #
            end = time.time()
            total_time += end - start

            if batch_idx % PRINT_PERIOD == 0:
                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]'
                    '\tBatch Neg ELBO: {:.6f}'
                    + 'Batch Neg IWELBO: {:.4f}')
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
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        end = time.time()
        total_time += end - start

        weight_w = decoder.layers[0].weight[[0]]
        weight_phi = encoder.fc1.weight[[0]]
        #train_data = torch.Tensor(train_loader.dataset)
        #neg_loglikelihood = toylosses.fa_neg_loglikelihood(
        #    weight_w, train_data)

        logging.info(
            '====> Epoch: {} '
            'Average Neg ELBO: {:.4f}'
            'Average Neg IWELBO: {:.4f}'.format(
                epoch, average_neg_elbo, average_neg_iwelbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = 0 #neg_loglikelihood
        train_losses['neg_elbo'] = average_neg_elbo
        train_losses['neg_iwelbo'] = average_neg_iwelbo
        train_losses['weight_w'] = weight_w
        train_losses['weight_phi'] = weight_phi
        train_losses['total_time'] = total_time

        return train_losses

    def val_vem(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_iwelbo = 0
        total_time = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        for batch_idx, batch_data in enumerate(val_loader):
            if DEBUG:
                if batch_idx > 3:
                    continue

            start = time.time()
            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = toynn.sample_from_q(
                mu, logvar, n_samples=N_MC_ELBO).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            # --- VEM: Neg ELBO
            half = int(n_batch_data / 2)
            batch_data_first_half = batch_data[:half, ]

            batch_data_expanded = batch_data_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)

            batch_recon_first_half = batch_recon[:half, ]
            batch_recon_expanded = batch_recon_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_recon_flat = batch_recon_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)

            batch_logvarx_first_half = batch_logvarx[:half, ]
            batch_logvarx_expanded = batch_logvarx_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_logvarx_flat = batch_logvarx_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon_flat, batch_logvarx_flat)

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
                n_is_samples=N_MC_TOT)
            end = time.time()
            total_time += end - start

            if batch_idx % PRINT_PERIOD == 0:
                string_base = (
                    'Val Epoch: {} [{}/{} ({:.0f}%)]'
                    '\tBatch Neg ELBO: {:.6f}'
                    + 'Batch Neg IWELBO: {:.4f}')
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
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        end = time.time()
        total_time += end - start

        weight = decoder.layers[0].weight[[0]]
        #val_data = torch.Tensor(val_loader.dataset)
        #neg_loglikelihood = toylosses.fa_neg_loglikelihood(
        #    weight, val_data)

        logging.info(
            '====> Val Epoch: {} '
            'Average Negative ELBO: {:.4f}'
            'Average IWAE: {:.4f}'.format(
                epoch, average_neg_elbo, average_neg_iwelbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = 0 #neg_loglikelihood
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['neg_iwelbo'] = average_neg_iwelbo
        val_losses['weight'] = weight
        val_losses['total_time'] = total_time
        return val_losses

    def run(self):
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        dataset_path = self.input().path
        dataset = torch.Tensor(np.load(dataset_path))

        logging.info('--Dataset tensor: (%d, %d)' % dataset.shape)

        n_train = int((1 - FRAC_VAL) * N_SAMPLES)
        train = torch.Tensor(dataset[:n_train, :])
        val = torch.Tensor(dataset[n_train:, :])

        logging.info('-- Train tensor: (%d, %d)' % train.shape)
        logging.info('-- Val tensor: (%d, %d)' % val.shape)

        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        val_dataset = torch.utils.data.TensorDataset(val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        m, o, s, t, v = train_utils.init_training(
            self.train_dir, NN_ARCHITECTURE, TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v
        for epoch in range(start_epoch, N_EPOCHS):
            if DEBUG:
                if epoch > 2:
                    break
            train_losses = self.train_vem(
                epoch, train_loader, modules, optimizers)
            train_losses_all_epochs.append(train_losses)
            val_losses = self.val_vem(
                epoch, val_loader, modules)
            val_losses_all_epochs.append(val_losses)

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
                self.models_path,
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
    models_path = os.path.join(TRAIN_VEGAN_DIR, 'models')
    train_losses_path = os.path.join(TRAIN_VEGAN_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(TRAIN_VEGAN_DIR, 'val_losses.pkl')

    def requires(self):
        return MakeDataSet()

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
            if DEBUG:
                if batch_idx > 3:
                    continue

            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = toynn.sample_from_q(
                    mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = toynn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, batch_logvarx_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
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
            z_from_prior = toynn.sample_from_prior(
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

            if batch_idx % PRINT_PERIOD == 0:
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

        weight = decoder.layers[0].weight[[0]]
        train_data = torch.Tensor(train_loader.dataset)
        neg_loglikelihood = toylosses.fa_neg_loglikelihood(
            weight, train_data)

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
            if DEBUG:
                if batch_idx > 3:
                    continue

            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = toynn.sample_from_q(
                    mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = toynn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, batch_logvarx_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_reconstruction.backward(retain_graph=True)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            n_from_prior = int(np.random.uniform(
                low=n_batch_data - n_batch_data / 4,
                high=n_batch_data + n_batch_data / 4))
            z_from_prior = toynn.sample_from_prior(
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

            if batch_idx % PRINT_PERIOD == 0:
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

        weight = decoder.layers[0].weight[[0]]
        val_data = torch.Tensor(val_loader.dataset)
        neg_loglikelihood = toylosses.fa_neg_loglikelihood(
            weight, val_data)

        logging.info('====> Val Epoch: {} Average loss: {:.4f}'.format(
                epoch, average_loss))
        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = neg_loglikelihood
        val_losses['total'] = average_loss
        return val_losses

    def run(self):
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        dataset_path = self.input().path
        dataset = torch.Tensor(np.load(dataset_path))

        logging.info('--Dataset tensor: (%d, %d)' % dataset.shape)

        n_train = int((1 - FRAC_VAL) * N_SAMPLES)
        train = torch.Tensor(dataset[:n_train, :])
        val = torch.Tensor(dataset[n_train:, :])

        logging.info('-- Train tensor: (%d, %d)' % train.shape)
        logging.info('-- Val tensor: (%d, %d)' % val.shape)

        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        val_dataset = torch.utils.data.TensorDataset(val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        m, o, s, t, v = train_utils.init_training(
            self.train_dir, NN_ARCHITECTURE, TRAIN_PARAMS)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v
        for epoch in range(start_epoch, N_EPOCHS):
            if DEBUG:
                if epoch > 2:
                    break
            train_losses = self.train_vegan(
                epoch, train_loader, modules, optimizers)
            train_losses_all_epochs.append(train_losses)
            val_losses = self.val_vegan(
                epoch, val_loader, modules)
            val_losses_all_epochs.append(val_losses)

            if epoch % CKPT_PERIOD == 0:
                train_utils.save_checkpoint(
                    epoch=epoch, modules=modules, optimizers=optimizers,
                    dir_path=self.train_dir,
                    train_losses_all_epochs=train_losses_all_epochs,
                    val_losses_all_epochs=val_losses_all_epochs,
                    nn_architecture=NN_ARCHITECTURE,
                    train_params=TRAIN_PARAMS)


def init():
    directories = [
        OUTPUT_DIR, SYNTHETIC_DIR,
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
            }
        })
