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

DATASET_NAME = 'cryo_exp'

HOME_DIR = '/gpfs/slac/cryo/fs1/u/nmiolane/results'
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


    def val(self, epoch, val_loader, modules,
            reconstructions=RECONSTRUCTIONS,
            regularizations=REGULARIZATIONS):

    def run(self):
        for directory in (self.train_dir, self.losses_path):
            if not os.path.isdir(directory):
                os.mkdir(directory)
                os.chmod(directory, 0o777)
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
