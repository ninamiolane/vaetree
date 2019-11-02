"""Data processing pipeline."""

import importlib
import logging
import numpy as np
import os
import random
import time

import ray
import ray.tune

import geomstats
import torch
import torch.autograd
import torch.optim
import torch.utils.data

import datasets
import toylosses
import toynn
import train_utils

import warnings
warnings.filterwarnings("ignore")

DATASET_NAME = 'synthetic'

DEBUG = True

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

SYNTHETIC_PARAMS = {
    'logvarx_true': -5,
    'n': 10000,
    'manifold_name': 's2',
    'w_true': W_TRUE,
    'b_true': B_TRUE,
    'nonlinearity': NONLINEARITY
    }

NN_ARCHITECTURE = {
    'nn_type': 'toy',
    'data_dim': 2,
    'latent_dim': 1,
    'n_decoder_layers': 2,
    'nonlinearity': NONLINEARITY,
    'with_biasx': True,
    'with_logvarx': False,
    'lovarx_true': SYNTHETIC_PARAMS['logvarx_true'],
    'with_biasz': True,
    'with_logvarz': True}

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
    'reconstruction_type': 'riem',
    'reconstructions': [],
    'regularizations': [],
    'algo_name': 'vae',
    'vae_type': 'gvae'}


if NN_ARCHITECTURE['with_logvarx']:
    assert len(W_TRUE) == NN_ARCHITECTURE['n_decoder_layers'] + 1, len(W_TRUE)
else:
    assert len(W_TRUE) == NN_ARCHITECTURE['n_decoder_layers']

# Train
FRAC_VAL = 0.2

PRINT_PERIOD = 16
N_EPOCHS = 31


class Train(ray.tune.Trainable):

    def _setup(self, config):

        synthetic_params = SYNTHETIC_PARAMS
        synthetic_params['dir'] = self.logdir
        synthetic_params['n'] = config.get('n')
        synthetic_params['logvarx_true'] = config.get('logvarx_true')
        synthetic_params['manifold_name'] = config.get('manifold_name')

        nn_architecture = NN_ARCHITECTURE
        nn_architecture['latent_dim'] = config.get('latent_dim')
        nn_architecture['logvarx_true'] = synthetic_params['logvarx_true']
        if not nn_architecture['with_logvarx']:
            assert nn_architecture['logvarx_true'] is not None

        train_params = TRAIN_PARAMS
        train_params['lr'] = config.get('lr')
        train_params['batch_size'] = config.get('batch_size')
        train_params['algo_name'] = config.get('algo_name')
        train_params['vae_type'] = config.get('vae_type')

        train_dataset, val_dataset = datasets.get_datasets(
            dataset_name=DATASET_NAME,
            nn_architecture=nn_architecture,
            train_params=train_params,
            synthetic_params=synthetic_params)

        print(
            'Train tensor: %s' % train_utils.get_logging_shape(train_dataset))
        print(
            'Val tensor: %s' % train_utils.get_logging_shape(val_dataset))

        train_dataset = torch.Tensor(train_dataset)
        val_dataset = torch.Tensor(val_dataset)
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

        self.synthetic_params = synthetic_params
        self.train_params = train_params
        self.nn_architecture = nn_architecture

    def batch_print(self,
                    batch_idx, n_batch_data, n_data, n_batches,
                    neg_elbo, loss_reconstruction, loss_regularization,
                    mode):
        # TODO(nina): Change print depending on algo_name
        epoch = self._iteration
        mode_string = 'Train' if mode == 'train' else 'Val'
        if batch_idx % PRINT_PERIOD == 0:
            string_base = (
                mode_string + ' Epoch: {} [{}/{} ({:.0f}%)]'
                + '\tBatch Neg ELBO: {:.6f}'
                + '\nReconstruction: {:.6f}, Regularization: {:.6f}')
            print(
                string_base.format(
                    epoch, batch_idx * n_batch_data, n_data,
                    100. * batch_idx / n_batches,
                    neg_elbo,
                    loss_reconstruction, loss_regularization))

    def batch_iteration(self, batch_data, algo_name, mode):
        if mode == 'train':
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

        if algo_name == 'vem':
            return self.batch_iteration_vem(batch_data, mode)

        encoder = self.modules['encoder']
        decoder = self.modules['decoder']

        mu, logvar = encoder(batch_data)

        z = toynn.sample_from_q(
            mu, logvar, n_samples=N_MC_TOT).to(DEVICE)
        batch_recon, batch_logvarx = decoder(z)

        n_batch_data = len(batch_data)
        batch_data_expanded = batch_data.expand(
            N_MC_TOT, n_batch_data, self.nn_architecture['data_dim'])
        batch_data_flat = batch_data_expanded.resize(
            N_MC_TOT*n_batch_data, self.nn_architecture['data_dim'])
        loss_reconstruction = toylosses.reconstruction_loss(
            batch_data_flat,
            batch_recon,
            batch_logvarx,
            reconstruction_type=self.train_params['reconstruction_type'],
            manifold_name=self.synthetic_params['manifold_name'])

        loss_regularization = toylosses.regularization_loss(
            mu, logvar)  # kld

        neg_iwelbo = toylosses.neg_iwelbo(
             decoder, batch_data, mu, logvar, n_is_samples=N_MC_TOT)

        if mode == 'train':
            if algo_name == 'vae':
                loss_reconstruction.backward(retain_graph=True)
                loss_regularization.backward()
            elif algo_name == 'iwae':
                neg_iwelbo.backward()

            self.optimizers['encoder'].step()
            self.optimizers['decoder'].step()

        return neg_iwelbo, loss_reconstruction, loss_regularization

    def batch_iteration_vem(self, batch_data, mode='train'):
        encoder = self.modules['encoder']
        decoder = self.modules['decoder']

        mu, logvar = encoder(batch_data)

        z = toynn.sample_from_q(
            mu, logvar, n_samples=N_MC_TOT).to(DEVICE)
        batch_recon, batch_logvarx = decoder(z)

        n_batch_data = len(batch_data)

        # --- VEM: Train encoder with NEG ELBO, proxy for KL --- #
        half = int(n_batch_data / 2)
        batch_data_first_half = batch_data[:half, ]

        batch_data_expanded = batch_data_first_half.expand(
            N_MC_ELBO, half, self.nn_architecture['data_dim'])
        batch_data_flat = batch_data_expanded.resize(
            N_MC_ELBO*half, self.nn_architecture['data_dim'])

        batch_recon_first_half = batch_recon[:half, ]
        batch_recon_expanded = batch_recon_first_half.expand(
            N_MC_ELBO, half, self.nn_architecture['data_dim'])
        batch_recon_flat = batch_recon_expanded.resize(
            N_MC_ELBO*half, self.nn_architecture['data_dim'])

        batch_logvarx_first_half = batch_logvarx[:half, ]
        batch_logvarx_expanded = batch_logvarx_first_half.expand(
            N_MC_ELBO, half, self.nn_architecture['data_dim'])
        batch_logvarx_flat = batch_logvarx_expanded.resize(
            N_MC_ELBO*half, self.nn_architecture['data_dim'])

        loss_reconstruction = toylosses.reconstruction_loss(
            batch_data_flat, batch_recon_flat, batch_logvarx_flat)

        loss_regularization = toylosses.regularization_loss(
            mu[:half, ], logvar[:half, ])  # kld

        if mode == 'train':
            loss_reconstruction.backward(retain_graph=True)
            loss_regularization.backward(retain_graph=True)
            self.optimizers['encoder'].step()

        # --- VEM: Train decoder with IWAE, proxy for NLL --- #
        batch_data_second_half = batch_data[half:, ]

        if mode == 'train':
            self.optimizers['decoder'].zero_grad()

        neg_iwelbo = toylosses.neg_iwelbo(
            decoder,
            batch_data_second_half,
            mu[half:, ], logvar[half:, ],
            n_is_samples=N_MC_IWELBO)

        if mode == 'train':
            # This also fills the encoder, but we do not step
            neg_iwelbo.backward()
            self.optimizers['decoder'].step()

        return neg_iwelbo, loss_reconstruction, loss_regularization

    def epoch_iteration(self, mode='train'):
        epoch = self._iteration
        algo_name = self.train_params['algo_name']

        loader = self.train_loader if mode == 'train' else self.val_loader

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        total_neg_elbo = 0
        total_neg_iwelbo = 0
        total_time = 0

        n_data = len(loader.dataset)
        n_batches = len(loader)
        for batch_idx, batch_data in enumerate(loader):
            if DEBUG and batch_idx > 3:
                continue
            start = time.time()

            batch_data = batch_data.to(DEVICE)
            n_batch_data = len(batch_data)

            neg_iwelbo, loss_recon, loss_regu = self.batch_iteration(
                batch_data, algo_name=algo_name, mode=mode)
            neg_elbo = loss_recon + loss_regu

            end = time.time()
            total_time += end - start

            self.batch_print(
                   batch_idx, n_batch_data, n_data, n_batches,
                   neg_elbo, loss_recon, loss_regu,
                   mode=mode)

            start = time.time()
            total_loss_reconstruction += (
                n_batch_data * loss_recon.item())
            total_loss_regularization += (
                n_batch_data * loss_regu.item())
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

        print('====> Epoch: {} Average Neg ELBO: {:.4f}'.format(
                epoch, average_neg_elbo))

        losses = {}
        losses['reconstruction'] = average_loss_reconstruction
        losses['regularization'] = average_loss_regularization
        losses['neg_loglikelihood'] = 0  # neg_loglikelihood
        losses['neg_elbo'] = average_neg_elbo
        losses['neg_iwelbo'] = average_neg_iwelbo
        losses['total_time'] = total_time
        return losses

    def _train_iteration(self):
        for module in self.modules.values():
            module.train()

        losses = self.epoch_iteration(mode='train')

        self.train_losses_all_epochs.append(losses)

    def _train(self):
        self._train_iteration()
        return self._test()

    def _test(self):
        for module in self.modules.values():
            module.eval()

        losses = self.epoch_iteration(mode='train')

        self.val_losses_all_epochs.append(losses)
        return {'average_neg_elbo': losses['neg_elbo']}

    def _save(self, checkpoint_dir=None):
        epoch = self._iteration

        if checkpoint_dir is None:
            checkpoint_dir = self.logdir

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
                algo_name=self.algo_name,
                module_name=module_name,
                epoch_id=epoch_id)


def init():
    logging.basicConfig(level=logging.INFO)
    print('start')


if __name__ == "__main__":
    init()

    ray.init()

    sched = ray.tune.schedulers.AsyncHyperBandScheduler(
        time_attr='training_iteration',
        metric='average_neg_elbo',
        mode='min',
        grace_period=N_EPOCHS-1)
    analysis = ray.tune.run(
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
            'num_samples': 1,
            # 'checkpoint_freq': CKPT_PERIOD,
            'checkpoint_at_end': True,
            'config': {
                'n': ray.tune.grid_search([1000, 5000]),  # , 10000, 100000]),
                'logvarx_true': ray.tune.grid_search(
                    [-10, -5]),   # , -3.22, -2, -1.02, -0.45, 0]),
                'manifold_name': ray.tune.grid_search(
                    ['s2', 'h2']),
                'algo_name': ray.tune.grid_search(
                    ['vae', 'iwae']),  # , 'vem']),
                'vae_type': ray.tune.grid_search(['gvae_tgt', 'vae']),
                'batch_size': TRAIN_PARAMS['batch_size'],
                'lr': TRAIN_PARAMS['lr'],
                'latent_dim': NN_ARCHITECTURE['latent_dim']
            }
        })
