"""Data processing pipeline."""

import jinja2
import logging
import luigi
import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os

import torch
import torch.autograd
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
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'training')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

DEBUG = True

CUDA = torch.cuda.is_available()
SEED = 12345
DEVICE = torch.device("cuda" if CUDA else "cpu")
torch.manual_seed(SEED)

# NN architecture
data_dim = 1
latent_dim = 1
n_layers = 1
nonlinearity = False
n_samples = 10000
with_biasx = False
with_logvarx = False

# True generative model
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

BATCH_SIZE = 64

PRINT_INTERVAL = 10
torch.backends.cudnn.benchmark = True

N_EPOCHS = 200

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
    output_path = os.path.join(SYNTHETIC_DIR, 'dataset.py')

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
    path = TRAIN_DIR
    models_path = os.path.join(TRAIN_DIR, 'models')
    losses_path = os.path.join(TRAIN_DIR, 'losses')
    train_losses_path = os.path.join(path, 'train_losses.pkl')

    def requires(self):
        return MakeDataSet()

    def toytrain_vae(self, epoch, train_loader, modules, optimizers):
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

        train_path = self.input()[train_name].path
        test_path = self.input()[test_name].path

        train = torch.Tensor(np.load(train_path))
        test = torch.Tensor(np.load(test_path))

        logging.info('-- Train tensor: (%d, %d, %d, %d)' % train.shape)
        np.random.shuffle(train)
        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        logging.info('-- Test tensor: (%d, %d, %d, %d)' % test.shape)
        test_dataset = torch.utils.data.TensorDataset(test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)



        test_losses_all_epochs.append(test_losses)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses_all_epochs, pkl)

        with open(self.output()['test_losses'].path, 'wb') as pkl:
            pickle.dump(test_losses_all_epochs, pkl)

    def output(self):
        return {'train_losses': luigi.LocalTarget(self.train_losses_path),
                'test_losses': luigi.LocalTarget(self.test_losses_path)}


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
        test_losses = {loss_type: [] for loss_type in loss_types}

        for i in epochs:
            path = os.path.join(TRAIN_DIR, 'losses', 'epoch_%d.pkl' % i)
            train_test = pickle.load(open(path, 'rb'))
            train = train_test['train']
            test = train_test['test']

            for loss_type in loss_types:
                loss = train[loss_type]
                train_losses[loss_type].append(loss)

                loss = test[loss_type]
                test_losses[loss_type].append(loss)

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
        plt.plot(test_losses['total'])
        plt.title('Test Loss')

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
        plt.plot(epochs, test_losses['discriminator'])
        plt.plot(epochs, test_losses['generator'])
        plt.plot(epochs, test_losses['reconstruction'])
        plt.plot(epochs, test_losses['regularization'])
        plt.title('Test Loss Decomposed')
        plt.legend(
            [loss_type for loss_type in loss_types if loss_type != 'total'],
            loc='upper right')

        # Only Discriminator and Generator
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
        plt.plot(epochs, test_losses['discriminator'])
        plt.plot(epochs, test_losses['generator'])
        plt.title('Test Loss: Discriminator and Generator only')
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
