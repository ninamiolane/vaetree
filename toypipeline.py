"""Data processing pipeline."""

import jinja2
import logging
import luigi
import numpy as np
import os
import pickle
import random

import torch
import torch.autograd
import torch.nn as tnn
from torch.nn import functional as F
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
TRAIN_VEM_DIR = os.path.join(OUTPUT_DIR, 'train_vem')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

DEBUG = True

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
data_dim = 1
latent_dim = 1
n_layers = 1
nonlinearity = False
with_biasx = False
with_logvarx = False

# True generative model
n_samples = 10000
w_true = {}
b_true = {}

# For the reconstruction
w_true[0] = [[2.]]
if with_biasx:
    b_true[0] = [[0.]]

if with_logvarx:
    # For the scale
    w_true[1] = [[0.]]
    b_true[1] = [[0.]]

# Train
FRAC_TEST = 0.2
BATCH_SIZE = 64
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

PRINT_INTERVAL = 10
torch.backends.cudnn.benchmark = True

N_EPOCHS = 200
LR = 1e-4

LATENT_DIM = 50
BETA1 = 0.5
BETA2 = 0.999


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

    def requires(self):
        pass

    def run(self):
        decoder_true = toynn.make_decoder_true(
            w_true, b_true, latent_dim, data_dim, n_layers,
            nonlinearity, with_biasx, with_logvarx)

        for name, param in decoder_true.named_parameters():
            print(name, param.data, '\n')

        synthetic_data = toynn.generate_from_decoder(decoder_true, n_samples)

        np.save(self.output().path, synthetic_data)

    def output(self):
        return luigi.LocalTarget(self.output_path)


class TrainVAE(luigi.Task):
    models_path = os.path.join(TRAIN_VAE_DIR, 'models')
    train_losses_path = os.path.join(TRAIN_VAE_DIR, 'train_losses.pkl')

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

            z = toynn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = toynn.sample_from_prior(
                    latent_dim, n_samples=n_batch_data).to(DEVICE)
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

            loss = loss_reconstruction + loss_regularization

            if batch_idx % PRINT_INTERVAL == 0:
                logloss = loss / n_batch_data
                logloss_reconstruction = loss_reconstruction / n_batch_data
                logloss_regularization = loss_regularization / n_batch_data

                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'
                    + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
                print(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        logloss,
                        logloss_reconstruction, logloss_regularization))

            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_regularization += loss_regularization.item()
            total_loss += loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_loss = total_loss / n_data

        print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, average_loss))
        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['total'] = average_loss
        return train_losses

    def run(self):
        for directory in (self.imgs_path, self.models_path, self.losses_path):
            if not os.path.isdir(directory):
                os.mkdir(directory)
                os.chmod(directory, 0o777)

        dataset_path = self.input().path
        dataset = torch.Tensor(np.load(dataset_path))

        logging.info('--Dataset tensor: (%d, %d)' % dataset.shape)

        n_train = int((1 - FRAC_TEST) * n_samples)
        train = torch.Tensor(dataset[:n_train, :])

        logging.info('-- Train tensor: (%d, %d)' % train.shape)
        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        # TODO(nina): Introduce test tensor

        vae = toynn.VAE(
            latent_dim=latent_dim, data_dim=data_dim,
            n_layers=n_layers, nonlinearity=nonlinearity,
            with_biasx=False,
            with_logvarx=False,
            with_biasz=False,
            with_logvarz=False)
        vae.to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

        print('\n-- Values of parameters before learning')
        decoder = modules['decoder']
        for name, param in decoder.named_parameters():
            print(name, param.data, '\n')

        optimizers = {}
        optimizers['encoder'] = torch.optim.Adam(
            modules['encoder'].parameters(), lr=LR, betas=(BETA1, BETA2))
        optimizers['decoder'] = torch.optim.Adam(
            modules['decoder'].parameters(), lr=LR, betas=(BETA1, BETA2))

        def init_xavier_normal(m):
            if type(m) == tnn.Linear:
                tnn.init.xavier_normal_(m.weight)
            else:
                print('Error of layer type.', type(m))

        for module in modules.values():
            module.apply(init_xavier_normal)

        train_losses_all_epochs = []

        for epoch in range(N_EPOCHS):
            train_losses = self.toytrain_vae(
                epoch, train_loader, modules, optimizers)
            train_losses_all_epochs.append(train_losses)

        for module_name, module in modules.items():
            module_path = os.path.join(
                self.models_path,
                '{}.pth'.format(module_name))
            torch.save(module, module_path)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses_all_epochs, pkl)

    def output(self):
        return {'train_losses': luigi.LocalTarget(self.train_losses_path)}


class TrainVEM(luigi.Task):
    models_path = os.path.join(TRAIN_VEM_DIR, 'models')
    train_losses_path = os.path.join(TRAIN_VEM_DIR, 'train_losses.pkl')

    def requires(self):
        return MakeDataSet()

    def train_vem(self, epoch, train_loader, modules, optimizers):
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
                    latent_dim, n_samples=n_batch_data).to(DEVICE)
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
                latent_dim, n_samples=n_from_prior)
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
            loss += loss_discriminator + loss_generator

            if batch_idx % PRINT_INTERVAL == 0:
                batch_loss = loss / n_batch_data
                batch_loss_reconstruction = loss_reconstruction / n_batch_data
                batch_loss_regularization = loss_regularization / n_batch_data
                batch_loss_discriminator = loss_discriminator / n_batch_data
                batch_loss_generator = loss_generator / n_batch_data

                dx = labels_data.mean()
                dgz = labels_from_prior.mean()

                string_base = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'
                    + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
                string_base += (
                    ', Discriminator: {:.6f}; Generator: {:.6f},'
                    '\nD(x): {:.3f}, D(G(z)): {:.3f}')
                print(
                    string_base.format(
                        epoch, batch_idx * n_batch_data, n_data,
                        100. * batch_idx / n_batches,
                        batch_loss,
                        batch_loss_reconstruction,
                        batch_loss_regularization,
                        batch_loss_discriminator,
                        batch_loss_generator,
                        dx, dgz))

            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_regularization += loss_regularization.item()
            total_loss_discriminator += loss_discriminator.item()
            total_loss_generator += loss_generator.item()

            total_loss += loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_loss = total_loss / n_data

        print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, average_loss))
        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['total'] = average_loss
        return train_losses

    def run(self):
        for directory in (self.imgs_path, self.models_path, self.losses_path):
            if not os.path.isdir(directory):
                os.mkdir(directory)
                os.chmod(directory, 0o777)

        dataset_path = self.input().path
        dataset = torch.Tensor(np.load(dataset_path))

        logging.info('--Dataset tensor: (%d, %d)' % dataset.shape)

        n_train = int((1 - FRAC_TEST) * n_samples)
        train = torch.Tensor(dataset[:n_train, :])

        logging.info('-- Train tensor: (%d, %d)' % train.shape)
        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        # TODO(nina): Introduce test tensor

        vae = toynn.VAE(
            latent_dim=latent_dim, data_dim=data_dim,
            n_layers=n_layers, nonlinearity=nonlinearity,
            with_biasx=False,
            with_logvarx=False,
            with_biasz=False,
            with_logvarz=False)
        vae.to(DEVICE)

        discriminator = toynn.Discriminator(data_dim=data_dim).to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

        modules['discriminator'] = discriminator

        print('\n-- Values of parameters before learning')
        decoder = modules['decoder']
        for name, param in decoder.named_parameters():
            print(name, param.data, '\n')

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

        for epoch in range(N_EPOCHS):
            train_losses = self.train_vem(
                epoch, train_loader, modules, optimizers)

            train_losses_all_epochs.append(train_losses)

        for module_name, module in modules.items():
            module_path = os.path.join(
                self.models_path,
                '{}.pth'.format(module_name))
            torch.save(module, module_path)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses_all_epochs, pkl)

    def output(self):
        return {'train_losses': luigi.LocalTarget(self.train_losses_path)}


class Report(luigi.Task):
    report_path = os.path.join(REPORT_DIR, 'report.html')

    def requires(self):
        return TrainVEM()

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
    for directory in [OUTPUT_DIR, TRAIN_VAE_DIR, REPORT_DIR]:
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
