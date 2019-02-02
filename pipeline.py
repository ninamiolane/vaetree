"""Data processing pipeline."""

import glob
import logging
import luigi
import matplotlib
matplotlib.use('Agg')  # NOQA
import os
import random
from joblib import Parallel, delayed
import nibabel
import nilearn
import numpy as np
import pickle
import skimage.transform
import sklearn.model_selection
import torch
import torch.autograd
import torch.utils.data
import torchvision

import nn

HOME_DIR = '/scratch/users/johmathe'
OUTPUT_DIR = os.path.join(HOME_DIR, 'output')

CUDA = torch.cuda.is_available()
SEED = 12345
DEVICE = torch.device("cuda" if CUDA else "cpu")
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
torch.manual_seed(SEED)

BATCH_SIZE = 256
PRINT_INTERVAL = 10
N_EPOCHS = 50

LATENT_DIM = 100

LR = 1e-4

N_INTENSITIES = 100000  # For speed-up
SUM_PIXEL_THRESHOLD = 40
IMAGE_SIZE = (64, 64)

TARGET = '/neuro/'


class FetchOpenNeuroDataset(luigi.Task):
    file_list_path = './datasets/openneuro_files.txt'
    target_dir = '/neuro/'

    def dl_file(self, path):
        path = path.strip()
        target_path = TARGET + os.path.dirname(path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        os.system("aws --no-sign-reques s3 cp  s3://openneuro.org/%s %s" %
                  (path, target_path))

    def requires(self):
        pass

    def run(self):
        with open('files') as f:
            all_files = f.readlines()

        Parallel(n_jobs=10)(delayed(self.dl_file)(f) for f in all_files)

    def output(self):
        return luigi.LocalTarget(self.target_dir)


def normalization(imgs):
    imgs = imgs.unsqueeze(1)
    print(imgs.size())
    logging.info(
        '-- Normalization of images intensities.')
    intensities = imgs.reshape((-1))[:N_INTENSITIES]
    intensities_without_0 = intensities[intensities > 2]

    n_imgs = imgs.shape[0]
    mean = (torch.mean(intensities_without_0),) * n_imgs
    std = (torch.std(intensities_without_0),) * n_imgs
    print('MEAN: %s' % str(mean))
    print('STD: %s' % str(std))
    imgs = torchvision.transforms.Normalize(mean, std)(imgs)

    return imgs, mean, std

def process_file(path, output, template):
    logging.info('loading and resizing image %s', path)
    img = nibabel.load(path)
    img = nilearn.image.resample_to_img(img, template)
    array = img.get_fdata()
    array = np.nan_to_num(array)
    volume_max = np.max(array)
    volume_min = np.min(array)
    volume_delta = volume_max - volume_min
    array = (array - volume_min) / volume_delta
    for k in range(30, array.shape[2]-20):
        img_slice = array[:, :, k]
        img = skimage.transform.resize(img_slice, IMAGE_SIZE)
        output.append(img)

class MakeDataSet(luigi.Task):
    train_path = os.path.join(OUTPUT_DIR, 'train.pkl')
    test_path = os.path.join(OUTPUT_DIR, 'test.pkl')
    first_slice = 28
    last_slice = 228
    test_fraction = 0.2

    def requires(self):
        return {'dataset': FetchOpenNeuroDataset()}

    def run(self):
        path = self.input()['dataset'].path
        filepaths = glob.glob(path + '**/*.nii.gz', recursive=True)
        random.shuffle(filepaths)
        filepaths = filepaths[:200]
        n_vols = len(filepaths)
        logging.info('----- Number of 3D images: %d' % n_vols)

        first_filepath = filepaths[0]
        first_img = nibabel.load(first_filepath)
        first_array = first_img.get_fdata()

        logging.info('----- First filepath: %s' % first_filepath)
        logging.info(
            '----- First volume shape: (%d, %d, %d)' % first_array.shape)

        logging.info(
            '-- Selecting 2D slices on dim 1 from slide %d to slice %d'
            % (self.first_slice, self.last_slice))

        imgs = []
        template = nibabel.load(
            '/neuro/ds000030/sub-10159/anat/sub-10159_T1w.nii.gz')

        Parallel(backend="threading",n_jobs=12)(delayed(process_file)(f, imgs, template) for f in filepaths)
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

        with open(self.output()['train'].path, 'wb') as train_pkl:
            pickle.dump(train, train_pkl)

        with open(self.output()['test'].path, 'wb') as test_pkl:
            pickle.dump(test, test_pkl)

    def output(self):
        return {'train': luigi.LocalTarget(self.train_path),
                'test': luigi.LocalTarget(self.test_path)}


class Train(luigi.Task):
    path = os.path.join(OUTPUT_DIR, 'training')
    train_losses_path = os.path.join(path, 'train_losses.pkl')
    test_losses_path = os.path.join(path, 'test_losses.pkl')

    def requires(self):
        return MakeDataSet()

    def train(self, epoch, train_loader, model, optimizer):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(DEVICE)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = nn.loss_function(recon_batch, data, mu, logvar)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

            if batch_idx % PRINT_INTERVAL == 0:
                logging.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data)))

        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / (len(train_loader.dataset))))

        train_loss /= len(train_loader.dataset)
        return train_loss

    def test(self, epoch, test_loader, model):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data[0].to(DEVICE)
                recon_batch, mu, logvar = model(data)
                test_loss += nn.loss_function(
                    recon_batch, data, mu, logvar).item()

                data_path = os.path.join(
                    self.path,
                    'imgs',
                    'epoch_{}_data.npy'.format(epoch))
                recon_path = os.path.join(
                    self.path,
                    'imgs',
                    'epoch_{}_recon.npy'.format(epoch))
                np.save(data_path, data.cpu().numpy())
                np.save(recon_path, recon_batch.data.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    def run(self):
        with open(self.input()['train'].path, 'rb') as train_pkl:
            train = pickle.load(train_pkl)

        logging.info(
            '----- Train tensor shape: (%d, %d, %d, %d)' % train.shape)
        train_dataset = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        with open(self.input()['test'].path, 'rb') as test_pkl:
            test = pickle.load(test_pkl)

        logging.info(
            '----- Test tensor shape: (%d, %d, %d, %d)' % test.shape)
        test_dataset = torch.utils.data.TensorDataset(test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        model = nn.VAE(
            n_channels=1,
            latent_dim=LATENT_DIM,
            w_in=train.shape[2],
            h_in=train.shape[3]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_losses = []
        test_losses = []
        for epoch in range(N_EPOCHS):
            train_loss = self.train(epoch, train_loader, model, optimizer)
            test_loss = self.test(epoch, test_loader, model)
            model_path = os.path.join(
                self.path,
                'models',
                'epoch_{}_train_loss_{:.4f}_test_loss_{:.4f}.pth'.format(
                    epoch, train_loss, test_loss))
            torch.save(model, model_path)

            train_test_path = os.path.join(
                self.path,
                'losses',
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


class RunAll(luigi.Task):
    def requires(self):
        return Train()

    def output(self):
        return luigi.LocalTarget('dummy')


def init():
    logging.basicConfig(level=logging.INFO)
    logging.info('start')
    luigi.run(
        main_task_cls=RunAll(),
        cmdline_args=[
            '--local-scheduler',
        ])


if __name__ == "__main__":
    init()
