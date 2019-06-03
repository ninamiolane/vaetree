"""Data processing pipeline."""

import datetime as dt
import glob
import jinja2
import logging
import luigi
import numpy as np
import os
import pickle
import random
import time
import skimage.transform

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
CATASTROPHE = False

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
IM_H = 28
IM_W = 28
DATA_DIM = 28 * 28  # MNIST size
LATENT_DIM = 30
CNN = False
BCE = False

# MC samples
N_VEM_ELBO = 1
N_VEM_IWELBO = 99
N_VAE = 1  # N_VEM_ELBO + N_VEM_IWELBO
N_IWAE = N_VEM_ELBO + N_VEM_IWELBO

# for IWELBO to estimate the NLL
N_MC_NLL = 5000

# Train

FRAC_VAL = 0.2
DATASET_NAME = 'cryo'

BATCH_SIZE = 20
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

PRINT_INTERVAL = 64
torch.backends.cudnn.benchmark = True

N_EPOCHS = 25
CKPT_PERIOD = 1
LR = 1e-4

BETA1 = 0.5
BETA2 = 0.999

N_BATCH_PER_EPOCH = 1e10

if DEBUG:
    N_EPOCHS = 10
    BATCH_SIZE = 16
    N_VEM_ELBO = 1
    N_VEM_IWELBO = 399
    N_VAE = 1
    N_IWAE = 400
    N_MC_NLL = 50
    CKPT_PERIOD = 1
    N_BATCH_PER_EPOCH = 1e10

# Report
LOADER = jinja2.FileSystemLoader('./templates/')
TEMPLATE_ENVIRONMENT = jinja2.Environment(
    autoescape=False,
    loader=LOADER)
TEMPLATE_NAME = 'report.jinja2'


def get_dataloaders(dataset_name=DATASET_NAME,
                    frac_val=FRAC_VAL, batch_size=BATCH_SIZE, kwargs=KWARGS):
    logging.info('Loading data from dataset: %s' % dataset_name)
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(
            '../data', train=True, download=True,
            transform=transforms.ToTensor())
    elif dataset_name == 'omniglot':
        dataset = datasets.Omniglot(
            '../data', download=True,
            transform=transforms.Compose(
                [transforms.Resize((IM_H, IM_W)), transforms.ToTensor()]))
    elif dataset_name == 'cryo':
        paths = glob.glob('/cryo/job40_vs_job034/*.pkl')
        all_datasets = []
        for path in paths:
            with open(path, 'rb') as pkl:
                data = pickle.load(pkl)
                dataset = data['ParticleStack']
                dataset = skimage.transform.resize(
                    dataset, (len(dataset), IM_H, IM_W))
                all_datasets.append(dataset)
        all_datasets = np.vstack([d for d in all_datasets])
        dataset = torch.Tensor(all_datasets)
    else:
        raise ValueError('Unknown dataset name: %s' % dataset_name)
    length = len(dataset)
    train_length = int((1 - frac_val) * length)
    val_length = int(length - train_length)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length])

    if dataset_name == 'mnist' or dataset_name == 'cryo':
        train_tensor = train_dataset.dataset.data[train_dataset.indices]
        val_tensor = val_dataset.dataset.data[val_dataset.indices]
        logging.info(
            '-- Train tensor: (%d, %d, %d)' % train_tensor.shape)
        logging.info(
            '-- Valid tensor: (%d, %d, %d)' % val_tensor.shape)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader


def save_checkpoint(epoch, modules, optimizers, dir_path,
                    train_losses_all_epochs, val_losses_all_epochs,
                    ckpt_period=CKPT_PERIOD):
    if epoch % ckpt_period == 0:
        checkpoint = {}
        for module_name in modules.keys():
            module = modules[module_name]
            optimizer = optimizers[module_name]
            checkpoint[module_name] = {
                'module_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
            checkpoint['epoch'] = epoch
            checkpoint['train_losses'] = train_losses_all_epochs
            checkpoint['val_losses'] = val_losses_all_epochs

        checkpoint_path = os.path.join(
            dir_path, 'epoch_%d_checkpoint.pth' % epoch)
        torch.save(checkpoint, checkpoint_path)


def init_xavier_normal(m):
    if type(m) == tnn.Linear:
        tnn.init.xavier_normal_(m.weight)


def init_training(models_path, modules, optimizers):
    """Initialization: Load ckpts or xavier normal init."""
    start_epoch = 0
    train_losses_all_epochs = []
    val_losses_all_epochs = []

    path_base = os.path.join(
        models_path, 'epoch_*_checkpoint.pth')
    ckpts = glob.glob(path_base)
    if len(ckpts) == 0:
        logging.info('No checkpoints found. Initializing with Xavier Normal.')
        for module in modules.values():
            module.apply(init_xavier_normal)
    else:
        ckpts_ids_and_paths = [
            (int(f.split('_')[2]), f) for f in ckpts]
        ckpt_id, ckpt_path = max(
            ckpts_ids_and_paths, key=lambda item: item[0])
        logging.info('Found checkpoints. Initializing with %s.' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        for module_name in modules.keys():
            module = modules[module_name]
            optimizer = optimizers[module_name]
            module_ckpt = ckpt[module_name]
            module.load_state_dict(module_ckpt['module_state_dict'])
            optimizer.load_state_dict(
                module_ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            train_losses_all_epochs = ckpt['train_losses']
            val_losses_all_epochs = ckpt['val_losses']

    return (modules, optimizers, start_epoch,
            train_losses_all_epochs, val_losses_all_epochs)


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
        total_neg_loglikelihood = 0
        total_time = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue
            start = time.time()

            if DATASET_NAME == 'cryo':
                batch_data = np.vstack([torch.unsqueeze(b, 0) for b in batch_data])
                batch_data = torch.Tensor(batch_data).to(DEVICE)
            else:
                batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)
            #logging.info('Values of true \'encoder\' parameters:')
            #for name, param in encoder.named_parameters():
            #    logging.info(name)
            #    logging.info(param.data)

            z = imnn.sample_from_q(
                mu, logvar, n_samples=N_VAE).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)
            if CATASTROPHE:
                print(
                    'norm(mu) = {:.4f}'
                    '\t norm(logvar) = {:.4f}'
                    '\t norm(z) = {:.4f}'
                    '\t norm(batch_recon) = {:.4f}'
                    '\t norm(batch_data) = {:.4f}'.format(
                        torch.norm(mu),
                        torch.norm(logvar),
                        torch.norm(z),
                        torch.norm(batch_recon),
                        torch.norm(batch_data)))

            # --- VAE: Train wrt Neg ELBO --- #
            batch_data = batch_data.view(-1, DATA_DIM)
            batch_data_expanded = batch_data.expand(
                N_VAE, n_batch_data, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_VAE*n_batch_data, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon, batch_logvarx, bce=BCE)
            loss_reconstruction.backward(retain_graph=True)

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

            # neg_iwelbo1 = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=1, bce=BCE)
            # neg_iwelbo100 = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=100, bce=BCE)
            # neg_iwelbo5000 = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=5000, bce=BCE)
            # print('Neg ELBO: {:.4f}\t Neg IWELBO1: {:.4f};   Neg IWELBO100: {:.4f};   Neg IWELBO5000: {:.4f}'.format(
            #     neg_elbo, neg_iwelbo1, neg_iwelbo100, neg_iwelbo5000))

            # neg_iwelbo = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=N_IWAE, bce=BCE)

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            # neg_loglikelihood = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL, bce=BCE)

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
            # total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
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

        logging.info('====> Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_neg_elbo))

        train_losses = {}
        train_losses['reconstruction'] = average_loss_reconstruction
        train_losses['regularization'] = average_loss_regularization
        train_losses['neg_loglikelihood'] = 0  # average_neg_loglikelihood
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
        total_neg_loglikelihood = 0
        total_time = 0

        n_data = len(val_loader.dataset)
        n_batches = len(val_loader)
        for batch_idx, batch_data in enumerate(val_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
                continue

            start = time.time()
            if DATASET_NAME == 'cryo':
                batch_data = np.vstack([torch.unsqueeze(b, 0) for b in batch_data])
                batch_data = torch.Tensor(batch_data).to(DEVICE)
            else:
                batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = imnn.sample_from_q(
                mu, logvar, n_samples=1).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            batch_data = batch_data.view(-1, DATA_DIM)
            batch_data_expanded = batch_data.expand(
                N_VAE, n_batch_data, DATA_DIM)
            batch_data_flat = batch_data_expanded.resize(
                N_VAE*n_batch_data, DATA_DIM)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data_flat, batch_recon, batch_logvarx, bce=BCE)

            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization
            end = time.time()
            total_time += end - start

            # neg_iwelbo = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=N_MC_TOT, bce=BCE)

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL, bce=BCE)

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
            # total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
            total_neg_loglikelihood += n_batch_data * neg_loglikelihood.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        average_neg_elbo = total_neg_elbo / n_data
        # average_neg_iwelbo = total_neg_iwelbo / n_data
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
        val_losses['neg_iwelbo'] = 0.  # average_neg_iwelbo
        val_losses['total_time'] = total_time
        return val_losses

    def run(self):
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        train_loader, val_loader = get_dataloaders()

        if CNN:
            vae = imnn.VAECNN(
                latent_dim=LATENT_DIM,
                im_h=IM_H,
                im_w=IM_W)
            vae.to(DEVICE)
        else:
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

        m, o, s, t, v = init_training(self.models_path, modules, optimizers)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        for epoch in range(start_epoch, N_EPOCHS):
            train_losses = self.train_vae(
                epoch, train_loader, modules, optimizers)
            val_losses = self.val_vae(
                epoch, val_loader, modules)

            train_losses_all_epochs.append(train_losses)
            val_losses_all_epochs.append(val_losses)

            save_checkpoint(
                epoch, modules, optimizers, self.models_path,
                train_losses_all_epochs, val_losses_all_epochs)

        for module_name, module in modules.items():
            module_path = os.path.join(
                self.models_path, '{}.pth'.format(module_name))
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
        total_neg_loglikelihood = 0
        total_time = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
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

            # z = imnn.sample_from_q(mu, logvar).to(DEVICE)
            # batch_recon, batch_logvarx = decoder(z)

            # batch_data = batch_data.view(-1, DATA_DIM)
            # loss_reconstruction = toylosses.reconstruction_loss(
            #     batch_data, batch_recon, batch_logvarx, bce=BCE)
            # loss_regularization = toylosses.regularization_loss(
            #     mu, logvar)  # kld

            # neg_elbo = loss_reconstruction + loss_regularization

            # --- IWAE: Train wrt IWAE --- #
            start = time.time()
            batch_data = batch_data.view(-1, DATA_DIM)
            neg_iwelbo = toylosses.neg_iwelbo(
                decoder,
                batch_data,
                mu, logvar,
                n_is_samples=N_IWAE, bce=BCE)

            # print('Anyway: neg_iwelbo pipeline = ', neg_iwelbo)
            if neg_iwelbo != neg_iwelbo or neg_iwelbo > 1e6:
                #print('mu = ', mu)
                #print('logvar = ', logvar)
                z = imnn.sample_from_q(mu, logvar).to(DEVICE)
                #print('z = ', z)
                batch_recon, batch_logvarx = decoder(z)
                #print('batch_logvarx = ', batch_logvarx)
                #print('batch_recon = ', batch_recon)
                print('neg_iwelbo pipeline = ', neg_iwelbo)
                neg_iwelbo_bis = toylosses.neg_iwelbo(
                    decoder,
                    batch_data,
                    mu, logvar,
                    n_is_samples=N_IWAE, bce=BCE)
                print('neg_iwelbo bis = ', neg_iwelbo_bis)
                neg_iwelbo_ter = toylosses.neg_iwelbo(
                    decoder,
                    batch_data,
                    mu, torch.zeros_like(mu),
                    n_is_samples=N_IWAE, bce=BCE)

                print('neg_iwelbo_ter = ', neg_iwelbo_ter)
                # neg_iwelbo = neg_iwelbo_ter  # HACK
                # raise ValueError()
            neg_iwelbo.backward()

            optimizers['encoder'].step()
            optimizers['decoder'].step()
            end = time.time()
            total_time += end - start
            # ---------------------------- #

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            # neg_loglikelihood = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL, bce=BCE)

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
            # total_loss_reconstruction += (
            #     n_batch_data * loss_reconstruction.item())
            # total_loss_regularization += (
            #     n_batch_data * loss_regularization.item())
            # total_neg_elbo += n_batch_data * neg_elbo.item()
            total_neg_iwelbo += n_batch_data * neg_iwelbo.item()
            # total_neg_loglikelihood +=n_batch_data * neg_loglikelihood.item()
            end = time.time()
            total_time += end - start

        start = time.time()
        # average_loss_reconstruction = total_loss_reconstruction / n_data
        # average_loss_regularization = total_loss_regularization / n_data
        # average_neg_elbo = total_neg_elbo / n_data
        average_neg_iwelbo = total_neg_iwelbo / n_data
        # average_neg_loglikelihood = total_neg_loglikelihood / n_data
        end = time.time()
        total_time += end - start

        logging.info('====> Epoch: {} Average NEG IWELBO: {:.4f}'.format(
                epoch, average_neg_iwelbo))

        train_losses = {}
        train_losses['reconstruction'] = 0.  # average_loss_reconstruction
        train_losses['regularization'] = 0.  # average_loss_regularization
        train_losses['neg_loglikelihood'] = 0.  # average_neg_loglikelihood
        train_losses['neg_iwelbo'] = average_neg_iwelbo
        train_losses['neg_elbo'] = 0.  # average_neg_elbo
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
            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)
            end = time.time()
            total_time += end - start

            z = imnn.sample_from_q(mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            batch_data = batch_data.view(-1, DATA_DIM)
            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx, bce=BCE)
            loss_regularization = toylosses.regularization_loss(
                mu, logvar)  # kld

            neg_elbo = loss_reconstruction + loss_regularization

            # --- IWAE --- #
            start = time.time()
            batch_data = batch_data.view(-1, DATA_DIM)
            neg_iwelbo = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_VEM_IWELBO, bce=BCE)
            end = time.time()
            total_time += end - start
            # ------------ #

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL, bce=BCE)

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
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        train_loader, val_loader = get_dataloaders()

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

        m, o, s, t, v = init_training(self.models_path, modules, optimizers)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        for epoch in range(start_epoch, N_EPOCHS):
            train_losses = self.train_iwae(
                epoch, train_loader, modules, optimizers)
            val_losses = self.val_iwae(
                epoch, val_loader, modules)

            train_losses_all_epochs.append(train_losses)
            val_losses_all_epochs.append(val_losses)

            save_checkpoint(
                epoch, modules, optimizers, self.models_path,
                train_losses_all_epochs, val_losses_all_epochs)

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
        total_neg_loglikelihood = 0
        total_time = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            if DEBUG and batch_idx > N_BATCH_PER_EPOCH:
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
                batch_data_flat, batch_recon_flat, batch_logvarx_flat, bce=BCE)
            if loss_reconstruction != loss_reconstruction or loss_reconstruction > 5e4:
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
                    batch_data_flat, batch_recon_flat, batch_logvarx_flat, bce=BCE)
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
                bce=BCE)
            # This also fills the encoder, but we do not step
            neg_iwelbo.backward()
            if neg_iwelbo != neg_iwelbo or neg_iwelbo > 5e4:
                print('mu = ', mu)
                print('logvar = ', logvar)
                z = imnn.sample_from_q(mu, logvar).to(DEVICE)
                print('z = ', z)
                batch_recon, batch_logvarx = decoder(z)
                print('batch_logvarx = ', batch_logvarx)
                print('batch_recon = ', batch_recon)
                print('neg_iwelbo pipeline = ', neg_iwelbo)
                neg_iwelbo_bis = toylosses.neg_iwelbo(
                    decoder,
                    batch_data,
                    mu, logvar,
                    n_is_samples=N_IWAE, bce=BCE)
                print('neg_iwelbo bis = ', neg_iwelbo_bis)
                neg_iwelbo_ter = toylosses.neg_iwelbo(
                    decoder,
                    batch_data,
                    mu, torch.zeros_like(mu),
                    n_is_samples=N_IWAE, bce=BCE)

                neg_iwelbo = neg_iwelbo_ter  # HACK
            optimizers['decoder'].step()
            # ----------------------------------------------------- #
            end = time.time()
            total_time += end - start

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            # neg_loglikelihood = toylosses.neg_iwelbo(
            #     decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL, bce=BCE)

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
            batch_data = batch_data[0].to(DEVICE)
            batch_data = batch_data.view(-1, DATA_DIM)
            n_batch_data = len(batch_data)

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = imnn.sample_from_q(
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
                batch_data_flat, batch_recon_flat, batch_logvarx_flat, bce=BCE)

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
                bce=BCE)
            end = time.time()
            total_time += end - start

            # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
            neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL, bce=BCE)

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
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)
            os.chmod(self.models_path, 0o777)

        train_loader, val_loader = get_dataloaders()

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

        m, o, s, t, v = init_training(self.models_path, modules, optimizers)
        modules, optimizers, start_epoch = m, o, s
        train_losses_all_epochs, val_losses_all_epochs = t, v

        for epoch in range(start_epoch, N_EPOCHS):
            train_losses = self.train_vem(
                epoch, train_loader, modules, optimizers)
            train_losses_all_epochs.append(train_losses)
            val_losses = self.val_vem(
                epoch, val_loader, modules)
            val_losses_all_epochs.append(val_losses)

            save_checkpoint(
                epoch, modules, optimizers, self.models_path,
                train_losses_all_epochs, val_losses_all_epochs)

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

            z = imnn.sample_from_q(
                    mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = imnn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, batch_logvarx_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx, bce=BCE)
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

        # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
        neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL, bce=BCE)

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

            z = imnn.sample_from_q(
                    mu, logvar).to(DEVICE)
            batch_recon, batch_logvarx = decoder(z)

            z_from_prior = imnn.sample_from_prior(
                    LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_from_prior, batch_logvarx_from_prior = decoder(
                    z_from_prior)

            loss_reconstruction = toylosses.reconstruction_loss(
                batch_data, batch_recon, batch_logvarx, bce=BCE)
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

        # Neg IW-ELBO is the estimator for NLL for a high N_MC_NLL
        neg_loglikelihood = toylosses.neg_iwelbo(
                decoder, batch_data, mu, logvar, n_is_samples=N_MC_NLL, bce=BCE)

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

        train_loader, val_loader = get_dataloaders()

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
