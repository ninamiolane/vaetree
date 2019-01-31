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

HOME_DIR = '/scratch/users/johmathe'
OUTPUT_DIR = os.path.join(HOME_DIR, 'output')

CUDA = torch.cuda.is_available()
SEED = 12345
DEVICE = torch.device("cuda" if CUDA else "cpu")
KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
torch.manual_seed(SEED)

BATCH_SIZE = 32
PRINT_INTERVAL = 10
N_EPOCHS = 10 #20

LATENT_DIM = 5

LR = 15e-6

N_INTENSITIES = 100000  # For speed-up


def normalization(imgs):
    logging.info(
        '-- Normalization of images intensities.')
    intensities = imgs.reshape((-1))[:N_INTENSITIES]
    intensities_without_0 = intensities[intensities > 2]

    plt.subplot(3, 1, 1)
    plt.hist(intensities, bins=100)

    plt.subplot(3, 1, 2)
    plt.hist(intensities_without_0, bins=100)

    n_imgs = imgs.shape[0]
    mean = (torch.mean(intensities_without_0),) * n_imgs
    std = (torch.std(intensities_without_0),) * n_imgs

    imgs = torchvision.transforms.Normalize(mean, std)(imgs)
    intensities_normalized = imgs.reshape((-1))[:N_INTENSITIES]

    plt.subplot(3, 1, 3)
    plt.hist(intensities_normalized, bins=100)
    #plt.savefig(f'{HOME_DIR}/outputs/plots/intensities.png')

    return imgs, mean, std


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
    mean_and_std_path = os.path.join(OUTPUT_DIR, 'mean_and_std.pkl')
    first_slice = 28
    last_slice = 228
    test_fraction = 0.2

    def requires(self):
        return {'dataset': FetchDataSet()}

    def run(self):
        path = self.input()['dataset'].path
        filepaths = glob.glob(path + '/ds000245/*T1w.nii.gz')
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
            for k in range(self.first_slice, self.last_slice):
                imgs.append(array[:, k, :])
        imgs = np.asarray(imgs)
        imgs = torch.Tensor(imgs)

        imgs, mean, std = normalization(imgs)
        mean_and_std = {'mean': mean, 'std': std}

        new_shape = (imgs.shape[0],) + (1,) + imgs.shape[1:]
        imgs = imgs.reshape(new_shape)

        n_imgs = imgs.shape[0]
        logging.info('----- Number of 2D images: %d' % n_imgs)
        logging.info(
            '----- First image shape: (%d, %d, %d)' % imgs[0].shape)
        logging.info('----- Training set shape: (%d, %d, %d, %d)' % imgs.shape)

        logging.info('-- Plot and save first image')
        plt.figure(figsize=[5, 5])
        first_img = imgs[0, :, :, 0]
        plt.imshow(first_img, cmap='gray')
        plt.savefig('./plots/first_image')

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

        with open(self.output()['mean_and_std'].path, 'wb') as pkl:
            pickle.dump(mean_and_std, pkl)

    def output(self):
        return {'train': luigi.LocalTarget(self.train_path),
                'test': luigi.LocalTarget(self.test_path),
                'mean_and_std': luigi.LocalTarget(self.mean_and_std_path)}


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

    def test(self, epoch, test_loader, model, mean_and_std):
        mean = mean_and_std['mean'][0]
        std = mean_and_std['std'][0]
        inv_mean = - mean / std
        inv_std = 1. / std

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data[0].to(DEVICE)
                recon_batch, mu, logvar = model(data)
                test_loss += nn.loss_function(
                    recon_batch, data, mu, logvar).item()

                # mean = (inv_mean,) * data.data.shape[0]
                # std = (inv_std,) * data.data.shape[0]
                # data_unnormalized = torchvision.transforms.Normalize(
                #     mean, std)(data.data)

                # mean = (inv_mean,) * recon_batch.data.shape[0]
                # std = (inv_std,) * recon_batch.data.shape[0]
                # recon_unnormalized = torchvision.transforms.Normalize(
                #     mean, std)(recon_batch.data)

                data_path = os.path.join(
                    self.path,
                    'imgs',
                    'Epoch_{}_data.npy'.format(epoch))
                recon_path = os.path.join(
                    self.path,
                    'imgs',
                    'Epoch_{}_recon.npy'.format(epoch))
                np.save(data_path, data.cpu().numpy())
                np.save(recon_path, recon_batch.data.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    def run(self):
        with open(self.input()['mean_and_std'].path, 'rb') as pkl:
            mean_and_std = pickle.load(pkl)

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
            test_loss = self.test(epoch, test_loader, model, mean_and_std)
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