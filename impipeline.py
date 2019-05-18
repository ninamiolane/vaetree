"""Data processing pipeline."""

import datetime as dt
import jinja2
import logging
import luigi
import numpy as np
import os
import pickle
import random
import time

import torch
import torch.autograd
from torch.nn import functional as F
import torch.nn as tnn
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms

import imnn
import toylosses
import toynn

import warnings
warnings.filterwarnings("ignore")

HOME_DIR = '/scratch/users/nmiolane'
OUTPUT_DIR = os.path.join(HOME_DIR, 'imoutput')
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
DATA_DIM = 784  # = 28 x 28, MNIST size
LATENT_DIM = 20

# MC samples for AVEM
N_MC_ELBO = 1
N_MC_IWELBO = 99

# for ELBO in case of VAE; IWELBO in case of IWAE
N_MC_TOT = N_MC_ELBO + N_MC_IWELBO

# Train

BATCH_SIZE = 128
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

PRINT_INTERVAL = 16
torch.backends.cudnn.benchmark = True

N_EPOCHS = 80
LR = 1e-3

BETA1 = 0.5
BETA2 = 0.999

if DEBUG:
    BATCH_SIZE = 4
    N_SAMPLES = 20
    N_MC_ELBO = 5
    N_MC_IWELBO = 3

# Report
LOADER = jinja2.FileSystemLoader('./templates/')
TEMPLATE_ENVIRONMENT = jinja2.Environment(
    autoescape=False,
    loader=LOADER)
TEMPLATE_NAME = 'report.jinja2'


class TrainVAE(luigi.Task):
    models_path = os.path.join(TRAIN_VAE_DIR, 'models')
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
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = imnn.sample_from_q(
                mu, logvar, n_samples=1).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            # --- VAE: Train wrt Neg ELBO --- #
            batch_data_expanded = batch_data.expand(
                1, n_batch_data, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                1*n_batch_data, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon, batch_logvarx)
            loss_reconstruction.backward(retain_graph=True)

            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld
            loss_regularization.backward()

            optimizers['encoder'].step()
            optimizers['decoder'].step()
            # ------------------------------- #

            neg_elbo = loss_reconstruction + loss_regularization

            end = time.time()
            total_time += end - start

            neg_iwelbo = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_TOT)

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

        neg_loglikelihood = 0.  # Placeholder for AIS

        logging.info('====> Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_neg_elbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = neg_loglikelihood
        train_losses['neg_elbo'] = average_neg_elbo
        train_losses['neg_iwelbo'] = average_neg_iwelbo
        train_losses['total_time'] = total_time
        return train_losses

    def val_vae(self, epoch, val_loader, modules):
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
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = imnn.sample_from_q(
                mu, logvar, n_samples=1).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            batch_data_expanded = batch_data.expand(
                1, n_batch_data, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                1*n_batch_data, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon, batch_logvarx)

            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization
            end = time.time()
            total_time += end - start

            neg_iwelbo = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_TOT)

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

        neg_loglikelihood = 0.

        logging.info('====> Val Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_neg_elbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = neg_loglikelihood
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['neg_iwelbo'] = average_neg_iwelbo
        val_losses['total_time'] = total_time
        return val_losses

    def run(self):
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        train_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        val_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data', train=False, transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        logging.info(
            '-- Train tensor: (%d, %d, %d)' % train_loader.dataset.data.shape)
        logging.info(
            '-- Valid tensor: (%d, %d, %d)' % val_loader.dataset.data.shape)

        vae = imnn.VAE(
            latent_dim=LATENT_DIM,
            data_dim=DATA_DIM)
        vae.to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

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
        pass

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
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)
            end = time.time()
            total_time += end - start

            z = imnn.sample_from_q(mu, logvar).to(DEVICE)
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

        neg_loglikelihood = 0.

        logging.info('====> Epoch: {} Average NEG IWELBO: {:.4f}'.format(
                epoch, average_neg_iwelbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = neg_loglikelihood
        train_losses['neg_iwelbo'] = average_neg_iwelbo
        train_losses['neg_elbo'] = average_neg_elbo
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
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)
            end = time.time()
            total_time += end - start

            z = imnn.sample_from_q(mu, logvar).to(DEVICE)
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
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        end = time.time()
        total_time += end - start

        neg_loglikelihood = 0.

        logging.info('====> Val Epoch: {} Average NEG IWELBO: {:.4f}'.format(
                epoch, average_neg_iwelbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = neg_loglikelihood
        val_losses['neg_iwelbo'] = average_neg_iwelbo
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['total_time'] = total_time
        return val_losses

    def run(self):
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        train_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        val_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data', train=False, transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        logging.info(
            '-- Train tensor: (%d, %d, %d)' % train_loader.dataset.data.shape)
        logging.info(
            '-- Valid tensor: (%d, %d, %d)' % val_loader.dataset.data.shape)

        vae = imnn.VAE(
            latent_dim=LATENT_DIM,
            data_dim=DATA_DIM)
        vae.to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

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
            if DEBUG:
                if batch_idx > 3:
                    continue

            start = time.time()
            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)
            batch_data = batch_data.view(n_batch_data, DATA_DIM)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = imnn.sample_from_q(
                mu, logvar, n_samples=N_MC_ELBO).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            # --- VEM: Train encoder with NEG ELBO, proxy for KL --- #
            half = int(n_batch_data / 2)
            batch_data_first_half = batch_data[:half, ]
            batch_recon_first_half = batch_recon[:half, ]
            batch_logvarx_first_half = batch_logvarx[:half, ]

            batch_data_expanded = batch_data_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)
            batch_recon_expanded = batch_recon_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_recon_flat = batch_recon_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)
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

            if batch_idx % PRINT_INTERVAL == 0:
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

        neg_loglikelihood = 0.

        logging.info(
            '====> Epoch: {} '
            'Average Neg ELBO: {:.4f}'
            'Average Neg IWELBO: {:.4f}'.format(
                epoch, average_neg_elbo, average_neg_iwelbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = neg_loglikelihood
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
        total_time = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        for batch_idx, batch_data in enumerate(val_loader):
            if DEBUG:
                if batch_idx > 3:
                    continue

            start = time.time()
            batch_data = batch_data[0].to(DEVICE)
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = imnn.sample_from_q(
                mu, logvar, n_samples=N_MC_ELBO).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            # --- VEM: Neg ELBO
            half = int(n_batch_data / 2)
            batch_data_first_half = batch_data[:half, ]
            batch_recon_first_half = batch_recon[:half, ]
            batch_logvarx_first_half = batch_logvarx[:half, ]

            batch_data_expanded = batch_data_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)
            batch_recon_expanded = batch_recon_first_half.expand(
                N_MC_ELBO, half, DATA_DIM)
            batch_recon_flat = batch_recon_expanded.resize(
                N_MC_ELBO*half, DATA_DIM)
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

            if batch_idx % PRINT_INTERVAL == 0:
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

        neg_loglikelihood = 0.

        logging.info(
            '====> Val Epoch: {} '
            'Average Negative ELBO: {:.4f}'
            'Average IWAE: {:.4f}'.format(
                epoch, average_neg_elbo, average_neg_iwelbo))

        val_losses = {}
        val_losses['reconstruction'] = average_loss_reconstruction
        val_losses['regularization'] = average_loss_regularization
        val_losses['neg_loglikelihood'] = neg_loglikelihood
        val_losses['neg_elbo'] = average_neg_elbo
        val_losses['neg_iwelbo'] = average_neg_iwelbo
        val_losses['total_time'] = total_time
        return val_losses

    def run(self):
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        train_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        val_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data', train=False, transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        logging.info(
            '-- Train tensor: (%d, %d, %d)' % train_loader.dataset.data.shape)
        logging.info(
            '-- Valid tensor: (%d, %d, %d)' % val_loader.dataset.data.shape)

        vae = imnn.VAE(
            latent_dim=LATENT_DIM,
            data_dim=DATA_DIM)
        vae.to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

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
            if DEBUG:
                if batch_idx > 3:
                    continue

            batch_data = batch_data[0].to(DEVICE)
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = imnn.sample_from_q(
                    mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = imnn.sample_from_prior(
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
            z_from_prior = imnn.sample_from_prior(
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
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = imnn.sample_from_q(
                    mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = imnn.sample_from_prior(
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
            z_from_prior = imnn.sample_from_prior(
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

        train_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        val_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data', train=False, transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        logging.info(
            '-- Train tensor: (%d, %d, %d)' % train_loader.dataset.data.shape)
        logging.info(
            '-- Valid tensor: (%d, %, %d)' % val_loader.dataset.data.shape)

        vae = imnn.VAE(
            latent_dim=LATENT_DIM,
            data_dim=DATA_DIM)
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
        return TrainVAE(), TrainIWAE(), TrainVEM()

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
