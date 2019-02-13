"""Data processing pipeline."""

import glob
import logging
import luigi
import matplotlib
matplotlib.use('Agg')  # NOQA
import os
import jinja2
from joblib import Parallel, delayed
import nibabel
import numpy as np
import pickle
import random
import skimage.transform
import sklearn.model_selection
import tempfile
import torch
import torch.autograd
import torch.nn as tnn
import torch.utils.data

import metrics
import nn

HOME_DIR = '/scratch/users/nmiolane'
# Change the output directory for new experiment
OUTPUT_DIR = os.path.join(HOME_DIR, 'output')
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'training')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

DEBUG = True

CUDA = torch.cuda.is_available()
SEED = 12345
DEVICE = torch.device("cuda" if CUDA else "cpu")
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
torch.manual_seed(SEED)

BATCH_SIZE = 64
PRINT_INTERVAL = 10
REGULARIZATION = 'kullbackleibler'

if REGULARIZATION == 'adversarial':
    REAL_LABELS = torch.full(
        (BATCH_SIZE,), nn.FAKE_LABEL, device=DEVICE)
    FAKE_LABELS = torch.full(
        (BATCH_SIZE,), nn.FAKE_LABEL, device=DEVICE)

N_EPOCHS = 200
if DEBUG:
    N_EPOCHS = 2

LATENT_DIM = 20

LR = 15e-6

IMAGE_SIZE = (128, 128)

TARGET = '/neuro/'

LOADER = jinja2.FileSystemLoader('./templates/')
TEMPLATE_ENVIRONMENT = jinja2.Environment(
    autoescape=False,
    loader=LOADER)
TEMPLATE_NAME = 'report.jinja2'


class FetchOpenNeuroDataset(luigi.Task):
    file_list_path = './datasets/openneuro_files.txt'
    target_dir = '/neuro/'

    def dl_file(self, path):
        path = path.strip()
        target_path = TARGET + os.path.dirname(path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        os.system("aws --no-sign-request s3 cp  s3://openneuro.org/%s %s" %
                  (path, target_path))

    def requires(self):
        pass

    def run(self):
        with open(self.file_list_path) as f:
            all_files = f.readlines()

        Parallel(n_jobs=10)(delayed(self.dl_file)(f) for f in all_files)

    def output(self):
        return luigi.LocalTarget(self.target_dir)


def is_diag(M):
    return np.all(M == np.diag(np.diagonal(M)))


def get_tempfile_name(some_id='def'):
    return os.path.join(
        tempfile.gettempdir(),
        next(tempfile._get_candidate_names()) + "_" + some_id + ".nii.gz")


def affine_matrix_permutes_axes(affine_matrix):
    mat = affine_matrix[:3, :3]
    if not is_diag(mat):
        logging.info('not diagonal, skipping')
        return True
    if np.any(mat < 0):
        logging.info('negative values, skipping')
        return True
    return False


def process_file(path, output):
    logging.info('loading and resizing image %s', path)
    img = nibabel.load(path)
    if affine_matrix_permutes_axes(img.affine):
        return

    array = img.get_fdata()
    array = np.nan_to_num(array)
    std = np.std(array.reshape(-1))

    array = array / std
    mean = np.mean(array.reshape(-1))
    # HACK Alert - This is a way to check if the backgound is a white noise.
    if mean > 1.0:
        print('mean too high: %s' % mean)
        return

    processed_file = get_tempfile_name()
    os.system('/usr/lib/ants/N4BiasFieldCorrection -i %s -o %s -s 6' %
              (path, processed_file))
    # Uncomment to skip N4 Bias Field Correction:
    # os.system('cp %s %s' % (path, processed_file))
    img = nibabel.load(processed_file)

    array = img.get_fdata()
    array = np.nan_to_num(array)
    std = np.std(array.reshape(-1))
    # No centering because we're using cross-entropy loss.
    # Another HACK ALERT - statisticians please intervene.
    array = array / (4 * std)
    z_size = array.shape[2]
    z_start = int(0.5 * z_size)
    z_end = int(0.85 * z_size)
    for k in range(z_start, z_end):
        img_slice = array[:, :, k]
        img = skimage.transform.resize(img_slice, IMAGE_SIZE)
        output.append(img)
    os.remove(processed_file)


class MakeDataSet(luigi.Task):
    train_path = os.path.join(OUTPUT_DIR, 'train.npy')
    test_path = os.path.join(OUTPUT_DIR, 'test.npy')
    first_slice = 28
    last_slice = 228
    test_fraction = 0.2

    def requires(self):
        return {'dataset': FetchOpenNeuroDataset()}

    def run(self):
        path = self.input()['dataset'].path
        filepaths = glob.glob(path + '**/*.nii.gz', recursive=True)
        random.shuffle(filepaths)
        n_vols = len(filepaths)
        logging.info('----- 3D images: %d' % n_vols)

        first_filepath = filepaths[0]
        first_img = nibabel.load(first_filepath)
        first_array = first_img.get_fdata()

        logging.info('----- First filepath: %s' % first_filepath)
        logging.info(
            '----- First volume shape: (%d, %d, %d)' % first_array.shape)

        logging.info(
            '-- Selecting 2D slices on dim 1 from slide %d to slice %d'
            % (self.first_slice, self.last_slice))

        if DEBUG:
            filepaths = filepaths[:16]

        imgs = []
        Parallel(
            backend="threading",
            n_jobs=4)(delayed(process_file)(f, imgs) for f in filepaths)
        imgs = np.asarray(imgs)
        imgs = torch.Tensor(imgs)

        new_shape = (imgs.shape[0],) + (1,) + imgs.shape[1:]
        imgs = imgs.reshape(new_shape)

        logging.info(
            '----- 2D images:'
            'training set shape: (%d, %d, %d, %d)' % imgs.shape)

        logging.info('-- Split into train and test sets')
        split = sklearn.model_selection.train_test_split(
            np.array(imgs), test_size=self.test_fraction, random_state=13)
        train, test = split
        train = torch.Tensor(train)
        test = torch.Tensor(test)

        np.save(self.output()['train'].path, train)
        np.save(self.output()['test'].path, test)

    def output(self):
        return {'train': luigi.LocalTarget(self.train_path),
                'test': luigi.LocalTarget(self.test_path)}


class Train(luigi.Task):
    path = TRAIN_DIR
    imgs_path = os.path.join(TRAIN_DIR, 'imgs')
    models_path = os.path.join(TRAIN_DIR, 'models')
    losses_path = os.path.join(TRAIN_DIR, 'losses')
    train_losses_path = os.path.join(path, 'train_losses.pkl')
    test_losses_path = os.path.join(path, 'test_losses.pkl')

    def requires(self):
        return MakeDataSet()

    def train(self, epoch, train_loader,
              modules, optimizers,
              regularization=REGULARIZATION):
        """
        - modules: a dict with the bricks of the model,
        eg. encoder, decoder, discriminator, depending on the architecture
        - optimizers: a dict with optimizers corresponding to each module.
        """
        for module in modules:
            module.train()

        total_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(DEVICE)
            n_data = len(data)

            for optimizer in optimizers:
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(data)
            z = nn.sample_from_q(mu, logvar).to(DEVICE)
            recon_batch, scale_b = decoder(z)

            loss_reconstruction = nn.reconstruction_loss(
                data, recon_batch, scale_b)

            if regularization == 'kullbackleibler':
                loss_regularization = nn.regularization_loss(mu, logvar)

            elif regularization == 'adversarial':
                discriminator = modules['discriminator']

                real_z = nn.sample_from_prior().to(DEVICE)
                real_recon_batch, real_scale_b = decoder(real_z)

                loss_regularization = self.regularization_adversarial(
                    discriminato=discriminator,
                    real_recon_batch=real_recon_batch,
                    fake_recon_batch=recon_batch)

            elif regularization == 'wasserstein':
                raise NotImplementedError(
                    'Wasserstein regularization not implemented.')
            else:
                raise NotImplementedError(
                    'Regularization not implemented.')

            loss = loss_reconstruction + loss_regularization

            loss.backward()

            total_loss += loss.item()

            for optimizer in optimizers:
                optimizer.step()

            # TODO(nina): Add logging for the different losses
            if batch_idx % PRINT_INTERVAL == 0:
                logging.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]'
                    '\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * n_data,
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / n_data))

        average_loss = total_loss / (len(train_loader.dataset))
        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, average_loss))

        return average_loss

    def test(self, epoch, test_loader, modules, regularization=REGULARIZATION):
        for module in modules:
            module.eval()
        total_test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data[0].to(DEVICE)

                encoder = modules['encoder']
                decoder = modules['decoder']

                mu, logvar = encoder(data)
                z = nn.sample_from_q(mu, logvar).to(DEVICE)
                recon_batch, scale_b = decoder(z)

                test_loss_reconstruction = nn.reconstruction_loss(
                    data, recon_batch, scale_b)

                if regularization == 'kullbackleibler':
                    test_loss_regularization = nn.regularization_loss(
                        mu, logvar)

                elif regularization == 'adversarial':
                    discriminator = modules['discriminator']
                    real_z = nn.sample_from_prior().to(DEVICE)
                    real_recon_batch, real_scale_b = decoder(real_z)

                    test_loss_regularization = self.regularization_adversarial(
                        discriminato=discriminator,
                        real_recon_batch=real_recon_batch,
                        fake_recon_batch=recon_batch)

                elif regularization == 'wasserstein':
                    raise NotImplementedError(
                        'Wasserstein regularization not implemented.')
                else:
                    raise NotImplementedError(
                        'Regularization not implemented.')

                test_loss = test_loss_reconstruction + test_loss_regularization

                total_test_loss += test_loss.item()

                data_path = os.path.join(
                    self.imgs_path, 'epoch_{}_data.npy'.format(epoch))
                recon_path = os.path.join(
                    self.imgs_path, 'epoch_{}_recon.npy'.format(epoch))
                real_recon_path = os.path.join(
                    self.imgs_path, 'epoch_{}_real_recon.npy'.format(epoch))

                np.save(data_path, data.cpu().numpy())
                np.save(recon_path, recon_batch.data.cpu().numpy())
                np.save(real_recon_path, real_recon_batch.data.cpu().numpy())

        average_test_loss = total_test_loss / len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(average_test_loss))
        return average_test_loss

    def regularization_adversarial(self,
                                   discriminator,
                                   real_recon_batch,
                                   fake_recon_batch):

        # discriminator - real
        predicted_labels_real = discriminator(real_recon_batch)
        loss_real = nn.gan_loss(
            predicted_labels=predicted_labels_real,
            true_labels=REAL_LABELS)

        # discriminator - fake
        predicted_labels_fake = discriminator(fake_recon_batch)
        loss_fake = nn.gan_loss(
            predicted_labels=predicted_labels_fake,
            true_labels=FAKE_LABELS)

        loss_discriminator = loss_real + loss_fake

        # generator/decoder - wants to fool the discriminator
        loss_generator = nn.gan_loss(
            predicted_labels=predicted_labels_fake,
            true_labels=REAL_LABELS)

        loss_regularization = loss_discriminator + loss_generator
        return loss_regularization

    def run(self):
        for directory in (self.imgs_path, self.models_path, self.losses_path):
            if not os.path.isdir(directory):
                os.mkdir(directory)
                os.chmod(directory, 0o777)

        train = np.load(self.input()['train'].path)
        test = np.load(self.input()['test'].path)
        train = torch.Tensor(train)
        test = torch.Tensor(test)

        logging.info(
            '----- Train tensor shape: (%d, %d, %d, %d)' % train.shape)
        np.random.shuffle(train)
        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        logging.info(
            '----- Test tensor shape: (%d, %d, %d, %d)' % test.shape)
        test_dataset = torch.utils.data.TensorDataset(test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        vae = nn.VAE(
            n_channels=1,
            latent_dim=LATENT_DIM,
            in_w=train.shape[2],
            in_h=train.shape[3]).to(DEVICE)

        modules = {}
        modules['encoder'] = vae.encoder
        modules['decoder'] = vae.decoder

        if REGULARIZATION == 'adversarial':
            discriminator = nn.Discriminator(
                latent_dim=LATENT_DIM,
                in_channels=1,
                in_w=train.shape[2],
                in_h=train.shape[3]).to(DEVICE)
            modules['discriminator'] = discriminator

        optimizers = {}
        optimizers['encoder'] = torch.optim.Adam(
            modules['encoder'].parameters(), lr=LR)
        optimizers['decoder'] = torch.optim.Adam(
            modules['decoder'].parameters(), lr=LR)

        if REGULARIZATION == 'adversarial':
            optimizers['discriminator'] = torch.optim.Adam(
                modules['discriminator'].parameters(), lr=LR)

        def init_normal(m):
            if type(m) == tnn.Linear:
                tnn.init.xavier_normal_(m.weight)
            if type(m) == tnn.Conv2d:
                tnn.init.xavier_normal_(m.weight)

        for module in modules:
            module.apply(init_normal)

        train_losses = []
        test_losses = []
        for epoch in range(N_EPOCHS):
            train_loss = self.train(
                epoch, train_loader, modules, optimizers, REGULARIZATION)
            test_loss = self.test(
                epoch, test_loader, modules, REGULARIZATION)

            for module_name, module in modules.items():
                module_path = os.path.join(
                    self.models_path,
                    'epoch_{}_{}_'
                    'train_loss_{:.4f}_test_loss_{:.4f}.pth'.format(
                        epoch, module_name, train_loss, test_loss))
                torch.save(module, module_path)

            train_test_path = os.path.join(
                self.losses_path,
                'epoch_{}.pkl'.format(
                    epoch, train_loss, test_loss))
            with open(train_test_path, 'wb') as pkl:
                pickle.dump(
                    {'train_loss': train_loss, 'test_loss': test_loss}, pkl)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        with open(self.output()['train_losses'].path, 'wb') as pkl:
            pickle.dump(train_losses, pkl)

        with open(self.output()['test_losses'].path, 'wb') as pkl:
            pickle.dump(test_losses, pkl)

    def output(self):
        return {'train_losses': luigi.LocalTarget(self.train_losses_path),
                'test_losses': luigi.LocalTarget(self.test_losses_path)}


class Report(luigi.Task):
    report_path = os.path.join(REPORT_DIR, 'report.html')

    def requires(self):
        return Train()

    def run(self):
        epoch_id = N_EPOCHS - 1

        data_path = os.path.join(
            TRAIN_DIR, 'imgs', 'epoch_%d_data.npy' % epoch_id)
        recon_path = os.path.join(
            TRAIN_DIR, 'imgs', 'epoch_%d_recon.npy' % epoch_id)
        data = np.load(data_path)
        recon = np.load(recon_path)

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
