"""Data processing pipeline."""

import datetime as dt
import jinja2
import logging
import luigi
import math
import numpy as np
import os
import pickle
import random

import torch
import torch.autograd
from torch.nn import functional as F
import torch.nn as tnn
import torch.optim
import torch.utils.data

import toylosses
import toynn

import warnings
warnings.filterwarnings("ignore")

HOME_DIR = '/scratch/users/nmiolane'
OUTPUT_DIR = os.path.join(HOME_DIR, 'toyoutput')
SYNTHETIC_DIR = os.path.join(OUTPUT_DIR, 'synthetic')
TRAIN_VAE_DIR = os.path.join(OUTPUT_DIR, 'train_vae')
TRAIN_IWAE_DIR = os.path.join(OUTPUT_DIR, 'train_iwae')
TRAIN_VEM_DIR = os.path.join(OUTPUT_DIR, 'train_vem')
TRAIN_VEGAN_DIR = os.path.join(OUTPUT_DIR, 'train_vegan')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

DEBUG = False

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# Seed
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# NN architecture
DATA_DIM = 1
LATENT_DIM = 1
N_DECODER_LAYERS = 1
NONLINEARITY = False
N_SAMPLES = 2000  # Number of synthetic data
N_IS_SAMPLES = 1000
WITH_BIASX = False
WITH_LOGVARX = False

W_TRUE = {}
B_TRUE = {}

W_TRUE[0] = [[2.]]

if WITH_LOGVARX:
    assert len(W_TRUE) == N_DECODER_LAYERS + 1, len(W_TRUE)
else:
    assert len(W_TRUE) == N_DECODER_LAYERS

WITH_BIASZ = True
WITH_LOGVARZ = True

# Train
FRAC_VAL = 0.2

BATCH_SIZE = 32
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

PRINT_INTERVAL = 16
torch.backends.cudnn.benchmark = True

N_EPOCHS = 500
LR = 1e-4

BETA1 = 0.5
BETA2 = 0.999

if DEBUG:
    BATCH_SIZE = 4
    N_SAMPLES = 20
    N_IS_SAMPLES = 3


# Report
LOADER = jinja2.FileSystemLoader('./templates/')
TEMPLATE_ENVIRONMENT = jinja2.Environment(
    autoescape=False,
    loader=LOADER)
TEMPLATE_NAME = 'report.jinja2'


class MakeDataSet(luigi.Task):
    """
    Generate synthetic dataset from a "true" decoder.
    """
    output_path = os.path.join(SYNTHETIC_DIR, 'dataset.npy')
    decoder_true_path = os.path.join(SYNTHETIC_DIR, 'decoder_true.pth')

    def requires(self):
        pass

    def run(self):
        logging.info('Configuration:')

        logging.info('DATA_DIM = %d' % DATA_DIM)
        logging.info('LATENT_DIM = %d' % LATENT_DIM)
        logging.info('N_DECODER_LAYERS = %d' % N_DECODER_LAYERS)

        logging.info('NONLINEARITY=%r' % NONLINEARITY)
        logging.info('WITH_BIASX=%r' % WITH_BIASX)
        logging.info('WITH_LOGVARX=%r' % WITH_LOGVARX)
        logging.info('WITH_BIASZ=%r' % WITH_BIASZ)
        logging.info('WITH_LOGVARZ=%r' % WITH_LOGVARZ)

        logging.info('N_SAMPLES=%d' % N_SAMPLES)

        logging.info('W_TRUE:')
        logging.info(W_TRUE)
        logging.info('B_TRUE:')
        logging.info(B_TRUE)

        decoder_true = toynn.make_decoder_true(
            w_true=W_TRUE, b_true=B_TRUE,
            latent_dim=LATENT_DIM, data_dim=DATA_DIM,
            n_layers=N_DECODER_LAYERS,
            nonlinearity=NONLINEARITY,
            with_biasx=WITH_BIASX, with_logvarx=WITH_LOGVARX)

        logging.info('Values of true \'decoder\' parameters:')
        for name, param in decoder_true.named_parameters():
            logging.info(name)
            logging.info(param.data)

        synthetic_data = toynn.generate_from_decoder(
            decoder=decoder_true, n_samples=N_SAMPLES)

        np.save(self.output().path, synthetic_data)
        torch.save(decoder_true, self.decoder_true_path)

    def output(self):
        return luigi.LocalTarget(self.output_path)


class TrainVAE(luigi.Task):
    models_path = os.path.join(TRAIN_VAE_DIR, 'models')
    train_losses_path = os.path.join(
        TRAIN_VAE_DIR, 'train_losses.pkl')
    val_losses_path = os.path.join(
        TRAIN_VAE_DIR, 'val_losses.pkl')

    def requires(self):
        return MakeDataSet()

    def train_vae(self, epoch, train_loader, modules, optimizers):
        for module in modules.values():
            module.train()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
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
            # print('mu=', mu)
            # print('logvar=', logvar)

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)
            # print('batch_recon=', batch_recon)
            # print('batch_logvarx=', batch_logvarx)

            z_from_prior = toynn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, scale_b_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_reconstruction.backward(retain_graph=True)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld
            loss_regularization.backward()

            optimizers['encoder'].step()
            optimizers['decoder'].step()

            if math.isnan(loss_reconstruction.item()):
                raise ValueError('Reconstruction loss on this batch is nan.')
            if math.isnan(loss_regularization.item()):
                raise ValueError('Regularization loss on this batch is nan.')
            # print('reconstruction', loss_reconstruction)
            # print('regularization', loss_regularization)
            loss = loss_reconstruction + loss_regularization

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Neg ELBO: {:.6f}'
                    + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        loss,
                        loss_reconstruction, loss_regularization))

            # Total losses are not averaged, only the sum of losses
            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_loss += n_batch_data * loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_loss = total_loss / n_data

        weight = decoder.layers[0].weight[[0]]
        train_data = torch.Tensor(train_loader.dataset)
        neg_loglikelihood = toylosses.fa_neg_loglikelihood(
            weight, train_data)

        logging.info('====> Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_loss))
        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = neg_loglikelihood
        train_losses['neg_elbo'] = average_loss
        return train_losses

    def val_vae(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
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
            # print('mu=', mu)
            # print('logvar=', logvar)

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)
            # print('batch_recon=', batch_recon)
            # print('batch_logvarx=', batch_logvarx)

            z_from_prior = toynn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, scale_b_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            if math.isnan(loss_reconstruction.item()):
                raise ValueError('Reconstruction loss on this batch is nan.')
            if math.isnan(loss_regularization.item()):
                raise ValueError('Regularization loss on this batch is nan.')
            # print('reconstruction', loss_reconstruction)
            # print('regularization', loss_regularization)
            loss = loss_reconstruction + loss_regularization

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Val Epoch: {} [{}/{} ({:.0f}%)]\tBatch Neg ELBO: {:.6f}'
                    + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        loss,
                        loss_reconstruction, loss_regularization))

            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_loss += n_batch_data * loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_loss = total_loss / n_data

        weight = decoder.layers[0].weight[[0]]
        val_data = torch.Tensor(val_loader.dataset)
        neg_loglikelihood = toylosses.fa_neg_loglikelihood(weight, val_data)

        logging.info('====> Val Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_loss))
        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = neg_loglikelihood
        val_losses['neg_elbo'] = average_loss
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

        vae = toynn.VAE(
            latent_dim=LATENT_DIM,
            data_dim=DATA_DIM,
            n_layers=N_DECODER_LAYERS,
            nonlinearity=NONLINEARITY,
            with_biasx=WITH_BIASX,
            with_logvarx=WITH_LOGVARX,
            with_biasz=WITH_BIASZ,
            with_logvarz=WITH_LOGVARZ)
        vae.to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

        logging.info('Values of VAE\'s decoder parameters before training:')
        decoder = modules['decoder']
        for name, param in decoder.named_parameters():
            logging.info(name)
            logging.info(param.data)

        optimizers = {}
        optimizers['encoder'] = torch.optim.Adam(
            modules['encoder'].parameters(), lr=LR, betas=(BETA1, BETA2))
        optimizers['decoder'] = torch.optim.Adam(
            modules['decoder'].parameters(), lr=LR, betas=(BETA1, BETA2))

        def init_xavier_normal(m):
            if type(m) == tnn.Linear:
                tnn.init.xavier_normal_(m.weight)

        for module in modules.values():
            module.apply(init_xavier_normal)

        train_losses_all_epochs = []
        val_losses_all_epochs = []

        for epoch in range(N_EPOCHS):
            if DEBUG:
                if epoch > 2:
                    break
            train_losses = self.train_vae(
                epoch, train_loader, modules, optimizers)
            val_losses = self.val_vae(
                epoch, val_loader, modules)

            train_losses_all_epochs.append(train_losses)
            val_losses_all_epochs.append(val_losses)

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


class TrainIWAE(luigi.Task):
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
        total_iwae = 0

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
            # print('mu=', mu)
            # print('logvar=', logvar)

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)
            # print('batch_recon=', batch_recon)
            # print('batch_logvarx=', batch_logvarx)

            z_from_prior = toynn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, scale_b_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization

            # We only propagate the IWAE
            iwae = toylosses.iw_vae_loss(
                batch_data, batch_recon, batch_logvarx, mu, logvar, z)

            iwae.backward()

            optimizers['encoder'].step()
            optimizers['decoder'].step()

            if math.isnan(loss_reconstruction.item()):
                raise ValueError('Reconstruction loss on this batch is nan.')
            if math.isnan(loss_regularization.item()):
                raise ValueError('Regularization loss on this batch is nan.')
            # print('reconstruction', loss_reconstruction)
            # print('regularization', loss_regularization)

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch IWAE: {:.6f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        iwae))

            # Total losses are not averaged, only the sum of losses
            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            total_iwae += n_batch_data * iwae.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_iwae = total_iwae / n_data

        weight = decoder.layers[0].weight[[0]]
        train_data = torch.Tensor(train_loader.dataset)
        neg_loglikelihood = toylosses.fa_neg_loglikelihood(
            weight, train_data)

        logging.info('====> Epoch: {} Average IWAE: {:.4f}'.format(
                epoch, average_iwae))
        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = neg_loglikelihood
        train_losses['iwae'] = average_iwae
        train_losses['neg_elbo'] = average_neg_elbo
        return train_losses

    def val_iwae(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_iwae = 0

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
            # print('mu=', mu)
            # print('logvar=', logvar)

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)
            # print('batch_recon=', batch_recon)
            # print('batch_logvarx=', batch_logvarx)

            z_from_prior = toynn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, scale_b_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization

            iwae = toylosses.iw_vae_loss(
                batch_data, batch_recon, batch_logvarx, mu, logvar, z)

            if math.isnan(loss_reconstruction.item()):
                raise ValueError('Reconstruction loss on this batch is nan.')
            if math.isnan(loss_regularization.item()):
                raise ValueError('Regularization loss on this batch is nan.')
            # print('reconstruction', loss_reconstruction)
            # print('regularization', loss_regularization)

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Val Epoch: {} [{}/{} ({:.0f}%)]\tBatch IWAE: {:.6f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        iwae))

            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            total_iwae += n_batch_data * iwae.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_iwae = total_iwae / n_data

        weight = decoder.layers[0].weight[[0]]
        val_data = torch.Tensor(val_loader.dataset)
        neg_loglikelihood = toylosses.fa_neg_loglikelihood(weight, val_data)

        logging.info('====> Val Epoch: {} Average IWAE: {:.4f}'.format(
                epoch, average_iwae))
        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = neg_loglikelihood
        val_losses['iwae'] = average_iwae
        val_losses['neg_elbo'] = average_neg_elbo
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

        vae = toynn.VAE(
            latent_dim=LATENT_DIM,
            data_dim=DATA_DIM,
            n_layers=N_DECODER_LAYERS,
            nonlinearity=NONLINEARITY,
            with_biasx=WITH_BIASX,
            with_logvarx=WITH_LOGVARX,
            with_biasz=WITH_BIASZ,
            with_logvarz=WITH_LOGVARZ)
        vae.to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

        logging.info('Values of IWAE\'s decoder parameters before training:')
        decoder = modules['decoder']
        for name, param in decoder.named_parameters():
            logging.info(name)
            logging.info(param.data)

        optimizers = {}
        optimizers['encoder'] = torch.optim.Adam(
            modules['encoder'].parameters(), lr=LR, betas=(BETA1, BETA2))
        optimizers['decoder'] = torch.optim.Adam(
            modules['decoder'].parameters(), lr=LR, betas=(BETA1, BETA2))

        def init_xavier_normal(m):
            if type(m) == tnn.Linear:
                tnn.init.xavier_normal_(m.weight)

        for module in modules.values():
            module.apply(init_xavier_normal)

        train_losses_all_epochs = []
        val_losses_all_epochs = []

        for epoch in range(N_EPOCHS):
            if DEBUG:
                if epoch > 2:
                    break
            train_losses = self.train_iwae(
                epoch, train_loader, modules, optimizers)
            val_losses = self.val_iwae(
                epoch, val_loader, modules)

            train_losses_all_epochs.append(train_losses)
            val_losses_all_epochs.append(val_losses)

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
        total_iwae = 0

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
            # print('mu=', mu)
            # print('logvar=', logvar)

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)
            # print('batch_recon=', batch_recon)
            # print('batch_logvarx=', batch_logvarx)

            # Update the encoder wrt the elbo,
            # proxy for the KL to the posterior
            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_reconstruction.backward(retain_graph=True)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld
            loss_regularization.backward()

            # The encoder is trained with ELBO
            optimizers['encoder'].step()

            # Update the decoder wrt the log-likelihood
            optimizers['decoder'].zero_grad()
            mu, logvar = encoder(batch_data)

            mu_expanded = mu.expand(N_IS_SAMPLES, n_batch_data, LATENT_DIM)
            mu_expanded_flat = mu_expanded.resize(
                N_IS_SAMPLES*n_batch_data, LATENT_DIM)

            logvar_expanded = logvar.expand(N_IS_SAMPLES, n_batch_data, -1)
            logvar_expanded_flat = logvar_expanded.resize(
                N_IS_SAMPLES*n_batch_data, LATENT_DIM)

            z_expanded_flat = toynn.sample_from_q(
                mu_expanded_flat, logvar_expanded_flat).to(DEVICE)
            z_expanded = z_expanded_flat.resize(
                N_IS_SAMPLES, n_batch_data, LATENT_DIM)

            batch_recon_expanded_flat, batch_logvarx_expanded_flat = decoder(
                z_expanded_flat)
            batch_recon_expanded = batch_recon_expanded_flat.resize(
                N_IS_SAMPLES, n_batch_data, DATA_DIM)
            batch_logvarx_expanded = batch_logvarx_expanded_flat.resize(
                N_IS_SAMPLES, n_batch_data, DATA_DIM)

            batch_data_expanded = batch_data.expand(
                N_IS_SAMPLES, n_batch_data, DATA_DIM)

            iwae = toylosses.iw_vae_loss(
                batch_data_expanded,
                batch_recon_expanded, batch_logvarx_expanded,
                mu_expanded, logvar_expanded,
                z_expanded)

            # The decoder is trained with IWAE
            iwae.backward()
            optimizers['decoder'].step()

            if math.isnan(loss_reconstruction.item()):
                raise ValueError('Reconstruction loss on this batch is nan.')
            if math.isnan(loss_regularization.item()):
                raise ValueError('Regularization loss on this batch is nan.')
            # print('reconstruction', loss_reconstruction)
            # print('regularization', loss_regularization)
            neg_elbo = loss_reconstruction + loss_regularization

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Neg ELBO: {:.6f}'
                    + 'Batch IWAE: {:.4f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        neg_elbo,
                        iwae))

            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            total_iwae += n_batch_data * iwae.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_iwae = total_iwae / n_data

        weight = decoder.layers[0].weight[[0]]
        train_data = torch.Tensor(train_loader.dataset)
        neg_loglikelihood = toylosses.fa_neg_loglikelihood(
            weight, train_data)

        logging.info(
            '====> Epoch: {} '
            'Average Neg ELBO: {:.4f}'
            'Average IWAE: {:.4f}'.format(
                epoch, average_neg_elbo, average_iwae))
        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = neg_loglikelihood
        train_losses['neg_elbo'] = average_neg_elbo
        train_losses['iwae'] = average_iwae

        return train_losses

    def val_vem(self, epoch, val_loader, modules):
        for module in modules.values():
            module.eval()
        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_iwae = 0

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
            # print('mu=', mu)
            # print('logvar=', logvar)

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)
            # print('batch_recon=', batch_recon)
            # print('batch_logvarx=', batch_logvarx)

            # Update the encoder wrt the elbo,
            # proxy for the KL to the posterior
            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx)
            loss_reconstruction.backward(retain_graph=True)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            # Decoder wrt the log-likelihood
            mu, logvar = encoder(batch_data)

            mu_expanded = mu.expand(N_IS_SAMPLES, n_batch_data, LATENT_DIM)
            mu_expanded_flat = mu_expanded.resize(
                N_IS_SAMPLES*n_batch_data, LATENT_DIM)

            logvar_expanded = logvar.expand(N_IS_SAMPLES, n_batch_data, -1)
            logvar_expanded_flat = logvar_expanded.resize(
                N_IS_SAMPLES*n_batch_data, LATENT_DIM)

            z_expanded_flat = toynn.sample_from_q(
                mu_expanded_flat, logvar_expanded_flat).to(DEVICE)
            z_expanded = z_expanded_flat.resize(
                N_IS_SAMPLES, n_batch_data, LATENT_DIM)

            batch_recon_expanded_flat, batch_logvarx_expanded_flat = decoder(
                z_expanded_flat)
            batch_recon_expanded = batch_recon_expanded_flat.resize(
                N_IS_SAMPLES, n_batch_data, DATA_DIM)
            batch_logvarx_expanded = batch_logvarx_expanded_flat.resize(
                N_IS_SAMPLES, n_batch_data, DATA_DIM)

            batch_data_expanded = batch_data.expand(
                N_IS_SAMPLES, n_batch_data, DATA_DIM)

            iwae = toylosses.iw_vae_loss(
                batch_data_expanded,
                batch_recon_expanded, batch_logvarx_expanded,
                mu_expanded, logvar_expanded,
                z_expanded)

            if math.isnan(loss_reconstruction.item()):
                raise ValueError('Reconstruction loss on this batch is nan.')
            if math.isnan(loss_regularization.item()):
                raise ValueError('Regularization loss on this batch is nan.')
            # print('reconstruction', loss_reconstruction)
            # print('regularization', loss_regularization)
            neg_elbo = loss_reconstruction + loss_regularization

            if batch_idx % PRINT_INTERVAL == 0:
                string_base = (
                    'Val Epoch: {} [{}/{} ({:.0f}%)]\tBatch Neg ELBO: {:.6f}'
                    + 'Batch IWAE: {:.4f}')
                logging.info(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        neg_elbo,
                        iwae))

            total_loss_reconstruction += (
                n_batch_data * loss_reconstruction.item())
            total_loss_regularization += (
                n_batch_data * loss_regularization.item())
            total_neg_elbo += n_batch_data * neg_elbo.item()
            total_iwae += n_batch_data * iwae.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_iwae = total_iwae / n_data

        weight = decoder.layers[0].weight[[0]]
        val_data = torch.Tensor(val_loader.dataset)
        neg_loglikelihood = toylosses.fa_neg_loglikelihood(
            weight, val_data)

        logging.info(
            '====> Val Epoch: {} '
            'Average Negative ELBO: {:.4f}'
            'Average IWAE: {:.4f}'.format(
                epoch, average_neg_elbo, average_iwae))
        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = neg_loglikelihood
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['iwae'] = average_iwae

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

        vae = toynn.VAE(
            latent_dim=LATENT_DIM,
            data_dim=DATA_DIM,
            n_layers=N_DECODER_LAYERS,
            nonlinearity=NONLINEARITY,
            with_biasx=WITH_BIASX,
            with_logvarx=WITH_LOGVARX,
            with_biasz=WITH_BIASZ,
            with_logvarz=WITH_LOGVARZ)
        vae.to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

        logging.info('Values of VAE\'s decoder parameters before training:')
        decoder = modules['decoder']
        for name, param in decoder.named_parameters():
            logging.info(name)
            logging.info(param.data)

        optimizers = {}
        optimizers['encoder'] = torch.optim.Adam(
            modules['encoder'].parameters(), lr=LR, betas=(BETA1, BETA2))
        optimizers['decoder'] = torch.optim.Adam(
            modules['decoder'].parameters(), lr=LR, betas=(BETA1, BETA2))

        def init_xavier_normal(m):
            if type(m) == tnn.Linear:
                tnn.init.xavier_normal_(m.weight)

        for module in modules.values():
            module.apply(init_xavier_normal)

        train_losses_all_epochs = []
        val_losses_all_epochs = []

        for epoch in range(N_EPOCHS):
            if DEBUG:
                if epoch > 2:
                    break
            train_losses = self.train_vem(
                epoch, train_loader, modules, optimizers)
            train_losses_all_epochs.append(train_losses)
            val_losses = self.val_vem(
                epoch, val_loader, modules)
            val_losses_all_epochs.append(val_losses)

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

        vae = toynn.VAE(
            latent_dim=LATENT_DIM, data_dim=DATA_DIM,
            n_layers=N_DECODER_LAYERS, nonlinearity=NONLINEARITY,
            with_biasx=WITH_BIASX,
            with_logvarx=WITH_LOGVARX,
            with_biasz=WITH_BIASZ,
            with_logvarz=WITH_LOGVARZ)
        vae.to(DEVICE)

        discriminator = toynn.Discriminator(data_dim=DATA_DIM).to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

        modules['discriminator'] = discriminator

        logging.info('Values of VEGAN\'s decoder parameters before training:')
        decoder = modules['decoder']
        for name, param in decoder.named_parameters():
            logging.info(name)
            logging.info(param.data)

        optimizers = {}
        optimizers['encoder'] = torch.optim.Adam(
            modules['encoder'].parameters(), lr=LR, betas=(BETA1, BETA2))
        optimizers['decoder'] = torch.optim.Adam(
            modules['decoder'].parameters(), lr=LR, betas=(BETA1, BETA2))

        optimizers['discriminator'] = torch.optim.Adam(
            modules['discriminator'].parameters(), lr=LR, betas=(BETA1, BETA2))

        def init_xavier_normal(m):
            if type(m) == tnn.Linear:
                tnn.init.xavier_normal_(m.weight)

        for module in modules.values():
            module.apply(init_xavier_normal)

        train_losses_all_epochs = []
        val_losses_all_epochs = []

        for epoch in range(N_EPOCHS):
            if DEBUG:
                if epoch > 2:
                    break
            train_losses = self.train_vegan(
                epoch, train_loader, modules, optimizers)
            train_losses_all_epochs.append(train_losses)
            val_losses = self.val_vegan(
                epoch, val_loader, modules)
            val_losses_all_epochs.append(val_losses)

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


class Report(luigi.Task):
    report_path = os.path.join(REPORT_DIR, 'report.html')

    def requires(self):
        return TrainVAE(), TrainIWAE(), TrainVEM(), TrainVEGAN()

    def get_last_epoch(self):
        # Placeholder
        epoch_id = N_EPOCHS - 1
        return epoch_id

    def get_loss_history(self):
        last_epoch = self.get_last_epoch()
        loss_history = []
        for epoch_id in range(last_epoch):
            path = os.path.join(
                TRAIN_VAE_DIR, 'losses', 'epoch_%d' % epoch_id)
            loss = np.load(path)
            loss_history.append(loss)
        return loss_history

    def load_data(self, epoch_id):
        data_path = os.path.join(
            TRAIN_VAE_DIR, 'imgs', 'epoch_%d_data.npy' % epoch_id)
        data = np.load(data_path)
        return data

    def load_recon(self, epoch_id):
        recon_path = os.path.join(
            TRAIN_VAE_DIR, 'imgs', 'epoch_%d_recon.npy' % epoch_id)
        recon = np.load(recon_path)
        return recon

    def load_from_prior(self, epoch_id):
        from_prior_path = os.path.join(
            TRAIN_VAE_DIR, 'imgs', 'epoch_%d_from_prior.npy' % epoch_id)
        from_prior = np.load(from_prior_path)
        return from_prior

    def run(self):
        pass

        with open(self.output().path, 'w') as f:
            template = TEMPLATE_ENVIRONMENT.get_template(TEMPLATE_NAME)
            html = template.render('')
            f.write(html)

    def output(self):
        return luigi.LocalTarget(self.report_path)


class RunAll(luigi.Task):
    def requires(self):
        return Report()

    def output(self):
        return luigi.LocalTarget('dummy')


def init():
    directories = [
        OUTPUT_DIR, SYNTHETIC_DIR,
        TRAIN_VAE_DIR, TRAIN_IWAE_DIR, TRAIN_VEM_DIR, # TRAIN_VEGAN_DIR,
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
