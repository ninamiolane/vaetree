"""Data processing pipeline."""

import glob
import logging
import luigi
import os

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pickle
import sklearn.model_selection
import torch
import torch.autograd
import torch.utils.data
import torchvision

import nn

HOME_DIR = '/scratch/users/nmiolane'
OUTPUT_DIR = os.path.join(HOME_DIR, 'output')

CUDA = torch.cuda.is_available()
SEED = 12345
DEVICE = torch.device("cuda" if CUDA else "cpu")
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
torch.manual_seed(SEED)

BATCH_SIZE = 2
PRINT_INTERVAL = 1
N_EPOCHS = 1

LATENT_DIM = 5

LR = 1e-4


class FetchDataSet(luigi.Task):
    path = os.path.join(HOME_DIR, 'dataset')

    def requires(self):
        pass

    def run(self):
        pass

    def output(self):
        return luigi.LocalTarget(self.path)


class MakeDataSet(luigi.Task):
    train_path = os.path.join(OUTPUT_DIR, 'train.pkl')
    test_path = os.path.join(OUTPUT_DIR, 'test.pkl')
    first_slice = 42
    last_slice = 162
    test_fraction = 0.2

    def requires(self):
        return {'dataset': FetchDataSet()}

    def run(self):
        path = self.input()['dataset'].path
        filepaths = glob.glob(path + '*/sub*T1w.nii.gz')
        n_vols = len(filepaths)
        logging.info('----- Number of 3D images: %d' % n_vols)

        first_filepath = filepaths[0]
        first_img = nib.load(first_filepath)
        first_array = first_img.get_data()

        logging.info('----- First filepath: %s' % first_filepath)
        logging.info(
            '----- First volume shape: (%d, %d, %d)' % first_array.shape)

        logging.info(
            '-- Selecting 2D slices on dim 1 from slide %d to slice %d'
            % (self.first_slice, self.last_slice))

        imgs = []
        for i in range(n_vols):
            img = nib.load(filepaths[i])
            array = img.get_data()
            print('!!!!!! array.shape = (%d, %d, %d)' % array.shape)
            for k in range(self.first_slice, self.last_slice):
                imgs.append(array[:, k, :])
        imgs = np.asarray(imgs)
        new_shape = (imgs.shape[0],) + (1,) + imgs.shape[1:]
        imgs = imgs.reshape(new_shape)

        n_imgs = imgs.shape[0]
        logging.info('----- Number of 2D images: %d' % n_imgs)
        logging.info(
            '----- First image shape: (%d, %d, %d)' % imgs[0].shape)
        logging.info('----- Training set shape: (%d, %d, %d, %d)' % imgs.shape)

        logging.info(
            '-- Normalization of images intensities between 0.0 and 1.0')
        intensity_min = np.min(imgs)
        intensity_max = np.max(imgs)
        imgs = (imgs - intensity_min) / (intensity_max - intensity_min)

        logging.info(
            '----- Check: Min: %f, Max: %f' % (np.min(imgs), np.max(imgs)))

        logging.info('-- Plot and save first image')
        plt.figure(figsize=[5, 5])
        first_img = imgs[0, :, :, 0]
        plt.imshow(first_img, cmap='gray')
        plt.savefig('./plots/first_image')

        logging.info('-- Split into train and test sets')
        split = sklearn.model_selection.train_test_split(
            imgs, test_size=self.test_fraction, random_state=13)
        train, test = split

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
                    'Epoch_{}_data.jpg'.format(epoch))
                recon_path = os.path.join(
                    self.path,
                    'imgs',
                    'Epoch_{}_recon.jpg'.format(epoch))
                torchvision.utils.save_image(
                    data.data,
                    data_path,
                    nrow=8,
                    padding=2)
                torchvision.utils.save_image(
                    recon_batch.data,
                    recon_path,
                    nrow=8,
                    padding=2)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    def run(self):
        with open(self.input()['train'].path, 'rb') as train_pkl:
            train = pickle.load(train_pkl)

        train_tensor = torch.Tensor(train)
        logging.info(
            '----- Train tensor shape: (%d, %d, %d, %d)' % train_tensor.shape)
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        with open(self.input()['test'].path, 'rb') as test_pkl:
            test = pickle.load(test_pkl)

        test_tensor = torch.Tensor(test)
        logging.info(
            '----- Test tensor shape: (%d, %d, %d, %d)' % test_tensor.shape)
        test_dataset = torch.utils.data.TensorDataset(test_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=True, **KWARGS)

        model = nn.VAE(n_channels=1, latent_dim=LATENT_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_losses = []
        test_losses = []
        for epoch in range(N_EPOCHS):
            train_loss = self.train(epoch, train_loader, model, optimizer)
            test_loss = self.test(epoch, test_loader, model)
            model_path = os.path.join(
                self.path,
                'models',
                'Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(
                    epoch, train_loss, test_loss))
            torch.save(model.state_dict(), model_path)

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
