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
from torch.nn import functional as F
import torch.optim
import torch.utils.data
import visdom

import losses
import metrics
import nn

HOME_DIR = '/scratch/users/nmiolane'

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
torch.backends.cudnn.benchmark = True

RECONSTRUCTIONS = ('adversarial', 'bce_on_intensities')
REGULARIZATIONS = ('kullbackleibler',)
WEIGHTS_INIT = 'kaiming'
REGU_FACTOR = 0.003

N_EPOCHS = 200
if DEBUG:
    N_EPOCHS = 2

LATENT_DIM = 20

LR = 15e-6
if 'adversarial' in RECONSTRUCTIONS:
    LR = 1e-6

REAL_LABELS = torch.full((BATCH_SIZE,), 1, device=DEVICE)
FAKE_LABELS = torch.full((BATCH_SIZE,), 0, device=DEVICE)

IMAGE_SIZE = (64, 64)

TARGET = '/neuro/'

LOADER = jinja2.FileSystemLoader('./templates/')
TEMPLATE_ENVIRONMENT = jinja2.Environment(
    autoescape=False,
    loader=LOADER)
TEMPLATE_NAME = 'report.jinja2'


method = 'original'


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
    first_slice = 118
    last_slice = 138
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

    def print_train_logs(self,
                         epoch,
                         batch_idx, n_batches, n_data, n_batch_data,
                         loss,
                         loss_reconstruction,
                         loss_regularization):

        loss = loss.item() / n_batch_data
        loss_reconstruction = loss_reconstruction.item() / n_batch_data
        loss_regularization = (
            loss_regularization.item() / n_batch_data)
        logging.info(
            'Train Epoch: {} [{}/{} ({:.0f}%)]'
            '\tLoss: {:.6f}'
            '\t(Reconstruction: {:.0f}%'
            ', Regularization: {:.0f}%)'.format(
                epoch,
                batch_idx * n_batch_data, n_data, 100. * batch_idx / n_batches,
                loss,
                100. * loss_reconstruction / loss,
                100. * loss_regularization / loss))

    def train(self, epoch, train_loader,
              modules, optimizers,
              reconstructions=RECONSTRUCTIONS,
              regularizations=REGULARIZATIONS):
        """
        - modules: a dict with the bricks of the model,
        eg. encoder, decoder, discriminator, depending on the architecture
        - optimizers: a dict with optimizers corresponding to each module.
        """
        for module in modules.values():
            module.train()

        total_loss_reconstruction = 0
        total_loss_regularization = 0
        if 'adversarial' in RECONSTRUCTIONS:
            total_loss_discriminator = 0
            total_loss_generator = 0
        total_loss = 0

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)

        vis = visdom.Visdom()
        vis.env = 'vae'
        data_win = None
        gen_win = None
        rec_win = None

        for batch_idx, batch_data in enumerate(train_loader):
            if DEBUG:
                if batch_idx > 1:
                    break
            batch_data = batch_data[0].to(DEVICE)
            n_batch_data = len(batch_data)
            assert n_batch_data == BATCH_SIZE

            data_win = vis.image(
                batch_data[0].cpu(),
                win = data_win,
                opts=dict(title='Real data'))

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            encoder = modules['encoder']
            decoder = modules['decoder']

            mu, logvar = encoder(batch_data)

            z = nn.sample_from_q(
                mu, logvar).to(DEVICE)
            batch_recon, scale_b = decoder(z)

            z_from_prior = nn.sample_from_prior(
                LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
            batch_recon_from_prior, scale_b_from_prior = decoder(
                z_from_prior)

            if 'adversarial' in reconstructions:
                # From:
                # Autoencoding beyond pixels using a learned similarity metric
                # arXiv:1512.09300v2
                discriminator = modules['discriminator_reconstruction']

                # -- Update Discriminator
                predicted_labels_data = discriminator(batch_data)
                predicted_labels_recon = discriminator(
                    batch_recon.detach())
                predicted_labels_recon_from_prior = discriminator(
                    batch_recon_from_prior.detach())

                loss_discriminator_data = F.binary_cross_entropy(
                    predicted_labels_data,
                    REAL_LABELS)
                loss_discriminator_recon = F.binary_cross_entropy(
                    predicted_labels_recon,
                    FAKE_LABELS)
                loss_discriminator_recon_from_prior = F.binary_cross_entropy(
                    predicted_labels_recon_from_prior,
                    FAKE_LABELS)

                # TODO(nina): add loss_discriminator_recon
                loss_discriminator = (
                    loss_discriminator_data
                    + loss_discriminator_recon_from_prior)

                # Fill gradients on discriminator only
                loss_discriminator.backward()

                # -- Update Generator/Decoder
                # Note that we need to do a forward pass with detached vars
                # in order not to propagate gradients through the encoder
                batch_recon_detached, _ = decoder(z.detach())
                # Note that we don't need to do it for batch_recon_from_prior
                # as it doesn't come from the encoder

                predicted_labels_recon = discriminator(
                    batch_recon_detached)
                predicted_labels_recon_from_prior = discriminator(
                    batch_recon_from_prior)

                loss_generator_recon = F.binary_cross_entropy(
                    predicted_labels_recon,
                    REAL_LABELS)

                # TODO(nina): add loss_generator_recon_from_prior
                loss_generator = loss_generator_recon

                # Fill gradients on generator only
                loss_generator.backward()

            if 'bce_on_intensities' in reconstructions:
                loss_reconstruction = losses.bce_on_intensities(
                    batch_data, batch_recon, scale_b)

                # Fill gradients on encoder and generator
                loss_reconstruction.backward(retain_graph=True)

            if 'kullbackleibler' in regularizations:
                loss_regularization = losses.kullback_leibler(mu, logvar)
                # Fill gradients on encoder only
                loss_regularization.backward()

            if 'adversarial' in regularizations:
                # From: Adversarial autoencoders
                # https://arxiv.org/pdf/1511.05644.pdf
                discriminator = modules['discriminator_regularization']
                raise NotImplementedError(
                    'Adversarial regularization not implemented.')

            if 'wasserstein' in regularizations:
                raise NotImplementedError(
                    'Wasserstein regularization not implemented.')

            optimizers['encoder'].step()
            optimizers['decoder'].step()
            optimizers['discriminator_reconstruction'].step()

            loss = loss_reconstruction + loss_regularization
            if 'adversarial' in RECONSTRUCTIONS:
                loss += loss_discriminator + loss_generator

            if batch_idx % PRINT_INTERVAL == 0:
                # TODO(nina): Implement print logs
                # of discriminator and generator
                if 'adversarial' in RECONSTRUCTIONS:
                    self.print_train_logs(
                        epoch,
                        batch_idx, n_batches, n_data, n_batch_data,
                        loss, loss_reconstruction, loss_regularization)
                else:
                    self.print_train_logs(
                        epoch,
                        batch_idx, n_batches, n_data, n_batch_data,
                        loss, loss_reconstruction, loss_regularization)

            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_regularization += loss_regularization.item()
            if 'adversarial' in RECONSTRUCTIONS:
                total_loss_discriminator += loss_discriminator.item()
                total_loss_generator += loss_generator.item()
            total_loss += loss.item()

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_loss_regularization = total_loss_regularization / n_data
        if 'adversarial' in RECONSTRUCTIONS:
            average_loss_discriminator = total_loss_discriminator / n_data
            average_loss_generator = total_loss_generator / n_data
        average_loss = total_loss / n_data

        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, average_loss))

        train_losses = {}
        train_losses['loss_reconstruction'] = average_loss_reconstruction
        train_losses['loss_regularization'] = average_loss_regularization
        if 'adversarial' in RECONSTRUCTIONS:
            train_losses['loss_discriminator'] = average_loss_discriminator
            train_losses['loss_generator'] = average_loss_generator
        train_losses['loss'] = average_loss
        return train_losses

    def test(self, epoch, test_loader, modules,
             reconstructions=RECONSTRUCTIONS,
             regularizations=REGULARIZATIONS):
        for module in modules.values():
            module.eval()

        total_loss = 0
        total_loss_reconstruction = 0
        total_weighted_loss_regularization = 0

        n_data = len(test_loader.dataset)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                if DEBUG:
                    if batch_idx > 1:
                        break
                batch_data = batch_data[0].to(DEVICE)
                n_batch_data = batch_data.shape[0]

                encoder = modules['encoder']
                decoder = modules['decoder']

                mu, logvar = encoder(batch_data)
                z = nn.sample_from_q(mu, logvar).to(DEVICE)
                batch_recon, scale_b = decoder(z)

                if 'bce_on_intensities' in reconstructions:
                    loss_reconstruction = losses.bce_on_intensities(
                        batch_data, batch_recon, scale_b)

                if 'adversarial' in reconstructions:
                    # From:
                    # Autoencoding beyond pixels using a learned
                    # similarity metric
                    # arXiv:1512.09300v2
                    discriminator = modules['discriminator_reconstruction']
                    loss_dis_real, loss_dis_fake_recon, _ = losses.adversarial(
                        discriminator=discriminator,
                        real_recon_batch=batch_data,
                        fake_recon_batch=batch_recon)

                    z_from_prior = nn.sample_from_prior(
                        LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
                    batch_recon_from_prior, scale_b_from_prior = decoder(
                        z_from_prior)

                    _, loss_dis_fake_from_prior, _ = losses.adversarial(
                        discriminator=discriminator,
                        real_recon_batch=batch_data,
                        fake_recon_batch=batch_recon_from_prior)

                    # TODO(nina): Add L2 norm on discriminator's activations
                    # discriminator_activations =

                    loss_reconstruction = (
                        loss_dis_real + loss_dis_fake_recon
                        + loss_dis_fake_from_prior)
                else:
                    raise NotImplementedError(
                        'This reconstruction loss is not implemented.')

                if 'kullbackleibler' in regularizations:
                    loss_regularization = losses.kullback_leibler(
                        mu, logvar)

                if 'adversarial' in regularizations:
                    discriminator = modules['discriminator_regularization']

                    z_from_prior = nn.sample_from_prior(
                        LATENT_DIM, n_samples=n_batch_data).to(DEVICE)
                    batch_recon_from_prior, scale_b_from_prior = decoder(
                        z_from_prior)

                    loss_real, loss_fake_from_prior, _ = losses.adversarial(
                        discriminator=discriminator,
                        real_recon_batch=batch_recon_from_prior,
                        fake_recon_batch=batch_recon)
                    loss_regularization = loss_real + loss_fake_from_prior

                if 'wasserstein' in regularizations:
                    raise NotImplementedError(
                        'Wasserstein regularization not implemented.')
                else:
                    raise NotImplementedError(
                        'This regularization loss is not implemented.')

                weighted_loss_regularization = (
                    REGU_FACTOR * loss_regularization)
                loss = loss_reconstruction + weighted_loss_regularization

                total_loss_reconstruction += loss_reconstruction.item()
                total_weighted_loss_regularization += (
                    weighted_loss_regularization.item())
                total_loss += loss.item()

                data_path = os.path.join(
                    self.imgs_path, 'epoch_{}_data.npy'.format(epoch))
                recon_path = os.path.join(
                    self.imgs_path, 'epoch_{}_recon.npy'.format(epoch))

                np.save(data_path, batch_data.data.cpu().numpy())
                np.save(recon_path, batch_recon.data.cpu().numpy())

                if 'adversarial' in regularizations:
                    recon_from_prior_path = os.path.join(
                        self.imgs_path,
                        'epoch_{}_recon_from_prior.npy'.format(epoch))
                    np.save(
                        recon_from_prior_path,
                        batch_recon_from_prior.data.cpu().numpy())

        average_loss_reconstruction = total_loss_reconstruction / n_data
        average_weighted_loss_regularization = (
            total_weighted_loss_regularization / n_data)
        average_loss = total_loss / n_data
        print('====> Test set loss: {:.4f}'.format(average_loss))

        test_losses = {}
        test_losses['loss_reconstruction'] = average_loss_reconstruction
        test_losses['weighted_loss_regularization'] = (
            average_weighted_loss_regularization)
        test_losses['loss'] = average_loss
        return test_losses

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

        if method == 'original':

            vae = nn.VAE(
                n_channels=1,
                latent_dim=LATENT_DIM,
                in_w=train.shape[2],
                in_h=train.shape[3]).to(DEVICE)

            modules = {}
            modules['encoder'] = vae.encoder
            modules['decoder'] = vae.decoder

            if 'adversarial' in RECONSTRUCTIONS:
                discriminator = nn.Discriminator(
                    latent_dim=LATENT_DIM,
                    in_channels=1,
                    in_w=train.shape[2],
                    in_h=train.shape[3]).to(DEVICE)
                modules['discriminator_reconstruction'] = discriminator

            if 'adversarial' in REGULARIZATIONS:
                discriminator = nn.Discriminator(
                    latent_dim=LATENT_DIM,
                    in_channels=1,
                    in_w=train.shape[2],
                    in_h=train.shape[3]).to(DEVICE)
                modules['discriminator_regularization'] = discriminator

            optimizers = {}
            optimizers['encoder'] = torch.optim.Adam(
                modules['encoder'].parameters(), lr=LR)
            optimizers['decoder'] = torch.optim.Adam(
                modules['decoder'].parameters(),
                lr=LR,
                betas=(nn.beta1, 0.999))

            if 'adversarial' in RECONSTRUCTIONS:
                optimizers['discriminator_reconstruction'] = torch.optim.Adam(
                    modules['discriminator_reconstruction'].parameters(),
                    lr=LR,
                    betas=(nn.beta1, 0.999))

            if 'adversarial' in REGULARIZATIONS:
                optimizers['discriminator_regularization'] = torch.optim.Adam(
                    modules['discriminator_regularization'].parameters(), lr=LR)

            def init_xavier_normal(m):
                if type(m) == tnn.Linear:
                    tnn.init.xavier_normal_(m.weight)
                if type(m) == tnn.Conv2d:
                    tnn.init.xavier_normal_(m.weight)

            def init_kaiming_normal(m):
                if type(m) == tnn.Linear:
                    tnn.init.kaiming_normal_(m.weight)
                if type(m) == tnn.Conv2d:
                    tnn.init.kaiming_normal_(m.weight)

            for module in modules.values():
                if WEIGHTS_INIT == 'xavier':
                    module.apply(init_xavier_normal)
                elif WEIGHTS_INIT == 'kaiming':
                    module.apply(init_kaiming_normal)
                else:
                    raise NotImplementedError(
                        'This weight initialization is not implemented.')

            train_losses_all_epochs = []
            test_losses_all_epochs = []
            for epoch in range(N_EPOCHS):
                train_losses = self.train(
                    epoch, train_loader, modules, optimizers,
                    RECONSTRUCTIONS, REGULARIZATIONS)
                #test_losses = self.test(
                #     epoch, test_loader, modules,
                #    RECONSTRUCTIONS, REGULARIZATIONS)

                #for module_name, module in modules.items():
                #    module_path = os.path.join(
                #        self.models_path,
                #        'epoch_{}_{}_'
                #        'train_loss_{:.4f}_test_loss_{:.4f}.pth'.format(
                #            epoch, module_name,
                #            train_losses['loss'], test_losses['loss']))
                #    torch.save(module, module_path)

                #train_test_path = os.path.join(
                #    self.losses_path, 'epoch_{}.pkl'.format(epoch))
                #with open(train_test_path, 'wb') as pkl:
                #    pickle.dump(
                #        {'train_losses': train_losses,
                #         'test_losses': test_losses},
                #        pkl)

                train_losses_all_epochs.append(train_losses)
                #test_losses_all_epochs.append(test_losses)

            with open(self.output()['train_losses'].path, 'wb') as pkl:
                pickle.dump(train_losses_all_epochs, pkl)

            #with open(self.output()['test_losses'].path, 'wb') as pkl:
            #    pickle.dump(test_losses_all_epochs, pkl)

        elif method == 'vaegan':
            vis = visdom.Visdom()
            vis.env = 'vae_dcgan'

            vis2 = visdom.Visdom()
            vis2.env = 'losses'
            loss_window = vis2.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros((1)).cpu(),
                opts=dict(xlabel='item',
                          ylabel='vae err',
                          title='vae err',
                          legend=['loss']))

            # custom weights initialization called on netG and netD
            def init_custom(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    m.weight.data.normal_(0.0, 0.02)
                elif classname.find('BatchNorm') != -1:
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)

            netG = nn._netG(nn.imageSize, nn.ngpu)  # image size, ngpu
            netG.apply(init_custom)
            print(netG)

            netD = nn._netD(nn.imageSize, nn.ngpu)
            netD.apply(init_custom)
            print(netD)

            criterion = torch.nn.BCELoss()
            MSECriterion = torch.nn.MSELoss()

            input = torch.FloatTensor(nn.batchSize, 3, nn.imageSize, nn.imageSize)
            noise = torch.FloatTensor(nn.batchSize, nn.nz, 1, 1)
            fixed_noise = torch.FloatTensor(nn.batchSize, nn.nz, 1, 1).normal_(0, 1)
            label = torch.FloatTensor(nn.batchSize)
            real_label = 1
            fake_label = 0

            if CUDA:
                netD.cuda()
                netG.make_cuda()
                criterion.cuda()
                MSECriterion.cuda()
                input, label = input.cuda(), label.cuda()
                noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

            input = torch.autograd.Variable(input)
            label = torch.autograd.Variable(label)
            noise = torch.autograd.Variable(noise)
            fixed_noise = torch.autograd.Variable(fixed_noise)

            # setup optimizer
            optimizerD = torch.optim.Adam(netD.parameters(), lr=nn.lr, betas=(nn.beta1, 0.999))
            optimizerG = torch.optim.Adam(netG.parameters(), lr=nn.lr, betas=(nn.beta1, 0.999))

            data_win = None
            gen_win = None
            rec_win = None

            for epoch in range(nn.niter):
                for i, data in enumerate(train_loader):
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    # train with real
                    netD.zero_grad()
                    real_cpu = data[0]

                    batch_size = real_cpu.size(0)
                    input.data.resize_(real_cpu.size()).copy_(real_cpu)
                    label.data.resize_(real_cpu.size(0)).fill_(real_label)

                    data_win = vis.image(
                        input.data[0].cpu(),
                        win = data_win,
                        opts=dict(title='Real data')
                        )

                    output = netD(input)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    D_x = output.data.mean()

                    # train with fake - decoding noise from prior
                    noise.data.resize_(batch_size, nn.nz, 1, 1)
                    noise.data.normal_(0, 1)
                    gen = netG.decoder(noise)
                    gen_win = vis.image(gen.data[0].cpu()*0.5+0.5, win = gen_win)

                    label.data.fill_(fake_label)
                    output = netD(gen.detach())
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.data.mean()
                    errD = errD_real + errD_fake
                    optimizerD.step()

                    ############################
                    # (2) Update G network: VAE
                    ###########################

                    netG.zero_grad()

                    encoded = netG.encoder(input)
                    mu = encoded[0]
                    logvar = encoded[1]

                    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                    KLD = torch.sum(KLD_element).mul_(-0.5)

                    sampled = netG.sampler(encoded)
                    rec = netG.decoder(sampled)
                    rec_win = vis.image(rec.data[0].cpu()*0.5+0.5,win = rec_win)

                    MSEerr = MSECriterion(rec, input)

                    # TODO(johmathe): Statistician needed. HACK alert.
                    VAEerr = (KLD + MSEerr)
                    VAEerr.backward()
                    optimizerG.step()

                    ############################
                    # (3) Update G network: maximize log(D(G(z)))
                    ###########################

                    label.data.fill_(real_label)  # fake labels are real for generator cost

                    rec = netG(input)  # this tensor is freed from mem at this point
                    output = netD(rec)
                    errG = criterion(output, label)
                    errG.backward()
                    D_G_z2 = output.data.mean()
                    optimizerG.step()

                    print('[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (epoch, nn.niter, i, len(train_loader),
                             VAEerr.data.item(), errD.data.item(), errG.data.item(), D_x, D_G_z1, D_G_z2))

                vis.line(X=torch.ones((1, 1)).cpu()*epoch,
                         Y=torch.Tensor([VAEerr.data.item()]).unsqueeze(0).cpu(),
                         win=loss_window,
                         update='append')

                torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (nn.outf, epoch))
                torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (nn.outf, epoch))


    def output(self):
        return {'train_losses': luigi.LocalTarget(self.train_losses_path)}
                #'test_losses': luigi.LocalTarget(self.test_losses_path)}


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
