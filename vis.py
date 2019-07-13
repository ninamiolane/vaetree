"""Visualization for experiments."""

import functools
import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pickle
import seaborn as sns
import torch

from matplotlib import animation
from scipy.stats import gaussian_kde

import analyze
import nn
import toylosses
import toynn


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


TIME_MAX = 600

TASK_TO_MARKER = {
    'rest': 'o',
    'breathhold': 'x',
    'dotstop': 'v',
    'eyesopen': '*',
    'languagewm': '^',
    'nback': '<',
    'objects': 'p',
    'retinotopy': 'P',
    'spatialwm': 'D'
}

SES_TO_MARKER = {
    2: 'o',
    3: 'x',
    4: 'v',
    5: 'D',
    10: 'P',
    11: 'p',
    12: 's',
    13: '*'
}

colormap = cm.get_cmap('viridis')
COLORS = colormap(np.linspace(start=0., stop=1., num=TIME_MAX))
ALPHA = 0.2
BINS = 40

ALGO_STRINGS = {
    'vae': 'VAE', 'iwae': 'IWAE', 'vem': 'AVEM'}
CRIT_STRINGS = {
    # From toypipeline and impipeline
    'neg_elbo': 'Neg ELBO',
    'neg_iwelbo': 'Neg IWELBO',
    'neg_loglikelihood': 'NLL',
    # From pipeline
    'total': 'Total',
    'reconstruction': 'Reconstruction',
    'regularization': 'Regularization',
    'discriminator': 'Discriminator',
    'generator': 'Generator'}
TRAIN_VAL_STRINGS = {'train': 'Train', 'val': 'Valid'}
COLOR_DICT = {
    'neg_elbo': 'red',
    'neg_iwelbo': 'orange',
    'total': 'lightblue',
    'reconstruction': 'lightgreen',
    'regularization': 'darkred',
    'discriminator': 'purple',
    'generator': 'violet'}
ALGO_COLOR_DICT = {'vae': 'red', 'iwae': 'orange', 'vem': 'blue'}
CMAPS_DICT = {'vae': 'Reds', 'iwae': 'Oranges', 'vem': 'Blues'}


FRAC_VAL = 0.2
N_SAMPLES = 10000
N_TRAIN = int((1 - FRAC_VAL) * N_SAMPLES)


def plot_data(x_data, color='darkgreen', label=None, ax=None):
    _, data_dim = x_data.shape
    if data_dim == 1:
        ax.hist(
            x_data, bins=BINS, alpha=ALPHA,
            color=color, label=label, density=True)
    else:
        sns.scatterplot(x_data[:, 0], x_data[:, 1], ax=ax, color=color)
    return ax


def plot_data_distribution(ax, output, algo_name='vae'):
    n_samples = 1000

    string_base = '%s/train_%s/decoder.pth' % (output, algo_name)
    decoder_path = glob.glob(string_base)[0]
    decoder = torch.load(decoder_path, map_location=DEVICE)

    string_base = '%s/synthetic/decoder_true.pth' % output
    decoder_true_path = glob.glob(string_base)[0]
    decoder_true = torch.load(decoder_true_path, map_location=DEVICE)

    generated_true_x = toynn.generate_from_decoder(decoder_true, n_samples)
    generated_x = toynn.generate_from_decoder(decoder, n_samples)

    plot_data(generated_true_x, color='darkgreen', ax=ax)
    plot_data(generated_x, color=ALGO_COLOR_DICT[algo_name], ax=ax)
    ax.set_title('Data distributions p(x)')
    ax.set_xlabel('x')
    return ax


def plot_posterior(ax, output, algo_name='vae'):
    n_to_sample = 10000
    w_true = 2
    x = 3

    string_base = '%s/train_%s/encoder.pth' % (output, algo_name)
    encoder_path = glob.glob(string_base)[0]
    encoder = torch.load(encoder_path, map_location=DEVICE)

    weight_phi = encoder.fc1.weight[[0]]
    weight_phi = weight_phi.detach().cpu().numpy()
    weight_phi = np.abs(weight_phi)

    true_loc = w_true / (w_true ** 2 + 1) * x
    true_scale = 1 / np.sqrt((1 + w_true ** 2))
    z_true_posterior = np.random.normal(
        loc=true_loc, scale=true_scale, size=(n_to_sample, 1))
    z_opt_posterior = np.random.normal(
        loc=true_loc, scale=1, size=(n_to_sample, 1))
    z_encoder = np.random.normal(
        loc=weight_phi*x, scale=1, size=(n_to_sample, 1))

    ax = plot_data(z_true_posterior, color='darkgreen', ax=ax)
    ax = plot_data(z_opt_posterior, color='blue', ax=ax)
    ax = plot_data(z_encoder, color=ALGO_COLOR_DICT[algo_name], ax=ax)
    ax.set_title('Posteriors z | x = 3')
    ax.set_xlabel('z')
    return ax


def plot_kde(some_x, ax, cmap, axis_side):
    x = some_x[:, 0]
    y = some_x[:, 1]
    data = np.vstack([x, y])
    kde = gaussian_kde(data)

    # evaluate on a regular grid
    xgrid = np.linspace(-axis_side, axis_side, 200)
    ygrid = np.linspace(-axis_side, axis_side, 200)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    # Plot the result as an image
    ax.imshow(
        Z.reshape(Xgrid.shape),
        origin='lower', aspect='auto',
        extent=[-2, 2, -2, 2],
        cmap=cmap)


def print_average_time(output, algo_name='vae'):
    string_base = '%s/train_%s/val_losses.pkl' % (output, algo_name)
    losses_path = glob.glob(string_base)[0]
    losses_all_epochs = pickle.load(open(losses_path, 'rb'))

    times = [loss['total_time'] for loss in losses_all_epochs]
    print(np.mean(times))


def print_weights(output, algo_name='vae', module_name='decoder'):
    string_base = '%s/train_%s/%s.pth' % (
        output, algo_name, module_name)
    module_path = glob.glob(string_base)[0]
    module = torch.load(module_path, map_location=DEVICE)

    print('\n-- Learnt values of parameters for module %s' % module_name)
    for name, param in module.named_parameters():
        print(name, param.data, '\n')


def plot_weights(ax, output, algo_name='vae',
                 start_epoch_id=0, epoch_id=None, color='blue', dashes=False):
    string_base = '%s/train_%s/train_losses.pkl' % (output, algo_name)
    losses_path = glob.glob(string_base)[0]
    losses_all_epochs = pickle.load(open(losses_path, 'rb'))

    weight_w = [loss['weight_w'] for loss in losses_all_epochs]
    weight_phi = [loss['weight_phi'] for loss in losses_all_epochs]

    # Take absolute value to avoid identifiability problem
    weight_w = np.abs(weight_w)
    weight_phi = np.abs(weight_phi)

    n_epochs = len(weight_w)
    epoch_id = min(epoch_id, n_epochs)

    label = '%s' % ALGO_STRINGS[algo_name]

    if not dashes:
        ax.plot(
            weight_w[start_epoch_id:epoch_id],
            weight_phi[start_epoch_id:epoch_id],
            label=label, color=color)
    else:
        ax.plot(
            weight_w[start_epoch_id:epoch_id],
            weight_phi[start_epoch_id:epoch_id],
            label=label, color=color, dashes=[2, 2, 2, 2])

    return ax


def load_checkpoint(output, algo_name='train_vae', epoch_id=None):
    if epoch_id is None:
        ckpts = glob.glob(
            '%s/train_%s/epoch_*_checkpoint.pth' % (
                output, algo_name))
        if len(ckpts) == 0:
            raise ValueError('No checkpoints found.')
        else:
            ckpts_ids_and_paths = [(int(f.split('_')[3]), f) for f in ckpts]
            ckpt_id, ckpt_path = max(
                ckpts_ids_and_paths, key=lambda item: item[0])
    else:
        # Load module corresponding to epoch_id
        ckpt_path = '%s/train_%s/epoch_%d_checkpoint.pth' % (
                output, algo_name, epoch_id)
        if not os.path.isfile(ckpt_path):
            raise ValueError('No checkpoints found for epoch %d.' % epoch_id)

    print('Found checkpoint. Getting: %s.' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    return ckpt


def load_module(output, algo_name='vae', module_name='encoder', epoch_id=None):
    print('Loading %s...' % module_name)
    ckpt = load_checkpoint(
        output=output, algo_name=algo_name, epoch_id=epoch_id)
    nn_architecture = ckpt['nn_architecture']

    nn_type = nn_architecture['nn_type']
    assert nn_type in ['linear', 'conv', 'gan']

    if nn_type == 'linear':
        vae = nn.Vae(
            latent_dim=nn_architecture['latent_dim'],
            data_dim=nn_architecture['data_dim'])
    elif nn_type == 'conv':
        vae = nn.VaeConv(
                latent_dim=nn_architecture['latent_dim'],
                img_shape=nn_architecture['img_shape'],
                spd=nn_architecture['spd'])
    else:
        vae = nn.VaeGan(
            latent_dim=nn_architecture['latent_dim'],
            img_shape=nn_architecture['img_shape'])
    vae.to(DEVICE)

    modules = {}
    modules['encoder'] = vae.encoder
    modules['decoder'] = vae.decoder
    module = modules[module_name]
    module_ckpt = ckpt[module_name]
    module.load_state_dict(module_ckpt['module_state_dict'])

    return module


def load_losses(output, algo_name='vae', epoch_id=None,
                crit_name='neg_elbo', mode='train'):
    ckpt = load_checkpoint(
        output=output, algo_name=algo_name, epoch_id=epoch_id)

    losses = ckpt['%s_losses' % mode]
    losses = [loss[crit_name] for loss in losses]

    return losses


def plot_criterion(ax, output, algo_name='vae', crit_name='neg_elbo',
                   mode='train', start_epoch_id=0, epoch_id=None,
                   color='blue', dashes=False):

    losses_total = load_losses(
        output=output, algo_name=algo_name,
        crit_name=crit_name, mode=mode, epoch_id=epoch_id)

    n_epochs = len(losses_total)
    epochs = range(n_epochs)
    if epoch_id is not None:
        epoch_id = min(epoch_id, n_epochs)

    label = '%s: %s %s' % (
        ALGO_STRINGS[algo_name],
        TRAIN_VAL_STRINGS[mode],
        CRIT_STRINGS[crit_name])

    if not dashes:
        ax.plot(
            epochs[start_epoch_id:epoch_id],
            losses_total[start_epoch_id:epoch_id],
            label=label, color=color)
    else:
        ax.plot(
            epochs[start_epoch_id:epoch_id],
            losses_total[start_epoch_id:epoch_id],
            label=label, color=color, dashes=[2, 2, 2, 2])
    ax.set_xlabel('epochs')
    ax.legend()
    return ax


def kl_posterior(w, phi, subdataset):

    sig2 = np.mean(subdataset ** 2, axis=0)

    kl_posterior_1 = - 1. / 2. * np.log(1 + w ** 2)
    kl_posterior_2 = 1. / 2. * (1 + w ** 2)
    kl_posterior_3 = (
        1. / 2. * (w - phi - phi * w ** 2) ** 2 * sig2 / (1 + w ** 2)
        - 1. / 2.)
    kl_posterior = kl_posterior_1 + kl_posterior_2 + kl_posterior_3
    return kl_posterior


def plot_kl_posterior(ax, output, algo_name='vae', mode='train', n_train=0,
                      start_epoch_id=0, epoch_id=None,
                      color='blue', dashes=False):

    string_base = '%s/train_%s/%s_losses.pkl' % (output, algo_name, mode)
    losses_path = glob.glob(string_base)[0]
    losses_all_epochs = pickle.load(open(losses_path, 'rb'))

    weight_w = [loss['weight_w'] for loss in losses_all_epochs]
    weight_phi = [loss['weight_phi'] for loss in losses_all_epochs]

    # Take absolute value to avoid identifiability problem
    weight_w = np.abs(weight_w)
    weight_phi = np.abs(weight_phi)

    n_epochs = len(weight_w)
    epochs = range(n_epochs)
    if epoch_id is not None:
        epoch_id = min(epoch_id, n_epochs)

    string_base = '%s/synthetic/dataset.npy' % output
    dataset_path = glob.glob(string_base)[0]
    dataset = np.load(dataset_path)
    if mode == 'train':
        subdataset = dataset[:n_train, ]
    if mode == 'val':
        subdataset = dataset[n_train:, ]

    kl = kl_posterior(weight_w, weight_phi, subdataset)
    label = '%s: %s KL' % (
        ALGO_STRINGS[algo_name], TRAIN_VAL_STRINGS[mode])

    if not dashes:
        ax.plot(epochs[start_epoch_id:epoch_id], kl[start_epoch_id:epoch_id],
                label=label, color=color)
    else:
        ax.plot(epochs[start_epoch_id:epoch_id], kl[start_epoch_id:epoch_id],
                label=label, color=color, dashes=[2, 2, 2, 2])
    ax.set_xlabel('epochs')
    return ax


def plot_kl_posterior_bis(ax, output, algo_name='vae', mode='train',
                          start_epoch_id=0, epoch_id=None, color='blue',
                          dashes=False):

    neg_elbo = load_losses(
        output=output, algo_name=algo_name,
        crit_name='neg_elbo', mode=mode)
    neg_ll = load_losses(
        output=output, algo_name=algo_name,
        crit_name='neg_loglikelihood', mode=mode)

    kl = [nelbo - nll for (nelbo, nll) in zip(neg_elbo, neg_ll)]

    n_epochs = len(kl)
    epochs = range(n_epochs)
    if epoch_id is not None:
        epoch_id = min(epoch_id, n_epochs)

    label = '%s: %s KL' % (
        ALGO_STRINGS[algo_name], TRAIN_VAL_STRINGS[mode])

    if not dashes:
        ax.plot(epochs[start_epoch_id:epoch_id], kl[start_epoch_id:epoch_id],
                label=label, color=color)
    else:
        ax.plot(epochs[start_epoch_id:epoch_id], kl[start_epoch_id:epoch_id],
                label=label, color=color, dashes=[2, 2, 2, 2])
    ax.set_xlabel('epochs')
    return ax


def plot_convergence(ax, output, algo_name, crit_name,
                     start_epoch_id=0, epoch_id=None):
    ax = plot_criterion(
        ax, output, algo_name=algo_name, crit_name=crit_name, mode='train',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id,
        color=COLOR_DICT[crit_name])
    ax = plot_criterion(
        ax, output, algo_name=algo_name, crit_name=crit_name, mode='val',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id,
        color=COLOR_DICT[crit_name], dashes=True)
    ax.set_title('Convergence of %s.' % ALGO_STRINGS[algo_name])
    return ax


def min_neg_ll(output, val=False):
    dataset_path = glob.glob(
        '%s/synthetic/dataset.npy' % output)[0]
    dataset = np.load(dataset_path)
    subdataset = torch.Tensor(dataset[:N_TRAIN, ])
    if val:
        subdataset = torch.Tensor(dataset[N_TRAIN:, ])

    w_mle_square = torch.mean(subdataset ** 2, dim=0) - 1
    w_mle = torch.sqrt(w_mle_square)
    w_mle = torch.Tensor(w_mle).to(DEVICE)

    min_neg_ll = toylosses.fa_neg_loglikelihood(w_mle, subdataset)

    min_neg_ll = min_neg_ll.cpu().numpy()
    w_mle = w_mle.cpu().numpy()

    return w_mle, min_neg_ll


def elbo_neg_ll(output, val='False'):
    dataset_path = glob.glob(
        '%s/synthetic/dataset.npy' % output)[0]
    dataset = np.load(dataset_path)
    subdataset = torch.Tensor(dataset[:N_TRAIN, ])
    if val:
        subdataset = torch.Tensor(dataset[N_TRAIN:, ])

    w_elbo_square = 0.5 * torch.mean(subdataset ** 2, dim=0) - 1
    w_elbo = torch.sqrt(w_elbo_square)

    w_elbo = torch.Tensor(w_elbo).to(DEVICE)

    neg_ll = toylosses.fa_neg_loglikelihood(w_elbo, subdataset)

    neg_ll = neg_ll.cpu().numpy()
    w_elbo = w_elbo.cpu().numpy()

    return w_elbo, neg_ll


def print_optimums(output):
    train_w_mle, train_min_neg_ll = min_neg_ll(output)
    val_w_mle, val_min_neg_ll = min_neg_ll(output, val=True)

    train_w_elbo, train_elbo_neg_ll = elbo_neg_ll(output)
    val_w_elbo, val_elbo_neg_ll = elbo_neg_ll(output, val=True)

    # Training set
    print('The maximum likelihood estimator on this train dataset is:')
    print(train_w_mle)

    print(
        'The corresponding value for the negative log-likelihood'
        ' for the training set is:')
    print(train_min_neg_ll)

    print('The w_elbo estimator on this train dataset is:')
    print(train_w_elbo)

    print(
        'The corresponding value for the negative log-likelihood'
        ' for the training set is:')
    print(train_elbo_neg_ll)

    print('\n')

    # Validation set

    print('The maximum likelihood estimator on this val dataset is:')
    print(val_w_mle)

    print(
        'The corresponding value for the negative log-likelihood'
        ' for the validation set is:')
    print(val_min_neg_ll)

    print('The w_elbo estimator on this validation dataset is:')
    print(val_w_elbo)

    print(
        'The corresponding value for the negative log-likelihood'
        ' for the validation set is:')
    print(val_elbo_neg_ll)


def plot_weights_landscape(ax, output, start_epoch_id=0, epoch_id=None):
    ax = plot_weights(
        ax, output, algo_name='vae',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id, color='red')
    ax = plot_weights(
        ax, output, algo_name='iwae',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id, color='orange')
    ax = plot_weights(
        ax, output, algo_name='vem',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id, color='blue')

    train_w_mle, _ = min_neg_ll()
    train_w_elbo, _ = elbo_neg_ll()

    w_opt = 2.
    phi_opt = w_opt / (1 + w_opt ** 2)

    w_mle = train_w_mle
    phi_mle = w_mle / (1 + w_mle ** 2)

    w_elbo = train_w_elbo
    phi_elbo = w_elbo / (1 + w_elbo ** 2)

    ax.scatter(
        w_opt, phi_opt, marker='o', s=80, color='darkgreen', label='True')
    ax.scatter(w_mle, phi_mle, marker='o', s=80, color='blue', label='MLE')
    ax.scatter(w_elbo, phi_elbo, marker='o', s=80, color='red', label='ELBO')

    ax.set_xlabel('w')
    ax.set_ylabel('phi')
    ax.legend()
    return ax


def plot_nll(ax, output, algo_name, start_epoch_id=0, epoch_id=None):
    ax = plot_criterion(
        ax, output, algo_name=algo_name, crit_name='neg_loglikelihood',
        mode='train', start_epoch_id=start_epoch_id, epoch_id=epoch_id,
        color=ALGO_COLOR_DICT[algo_name])
    ax = plot_criterion(
        ax, output, algo_name=algo_name, crit_name='neg_loglikelihood',
        mode='val', start_epoch_id=start_epoch_id, epoch_id=epoch_id,
        color=ALGO_COLOR_DICT[algo_name], dashes=True)
    return ax


def plot_kl(ax, output, algo_name, start_epoch_id=0, epoch_id=None):
    ax = plot_kl_posterior(
        ax, output, algo_name=algo_name, mode='train',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id,
        color=ALGO_COLOR_DICT[algo_name], dashes=False)
    ax = plot_kl_posterior(
        ax, output, algo_name=algo_name, mode='val',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id,
        color=ALGO_COLOR_DICT[algo_name], dashes=True)

    return ax


def play_fmri(array_4d):
    mat = array_4d
    slice_2d = 26

    def quick_play(dT=50):
        fig, ax = plt.subplots()
        im = ax.imshow(mat[:, :, slice_2d, 0], cmap='viridis')

        def init():
            im.set_data(mat[:, :, slice_2d, 0])
            return im,

        def animate(i):
            im.set_data(mat[:, :, slice_2d, i])
            return im,

        anima = animation.FuncAnimation(
            fig, animate, frames=array_4d.shape[3]-1,
            init_func=init, interval=dT, blit=True)
        return anima

    anima = quick_play()
    # HTML(anima.to_html5_video())


def show_data(filename, nrows=4, ncols=18, figsize=(18, 4), cmap='gray'):
    print('Loading %s' % filename)
    dataset = np.load(filename)
    print('Dataset shape:', dataset.shape)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize)
    n_samples = nrows * ncols

    for i_img, one_img in enumerate(dataset):
        if i_img > n_samples - 1:
            break
        if len(one_img.shape) == 3:
            one_img = one_img[0]  # channels
        ax = axes[int(i_img // ncols), int(i_img % ncols)]
        ax.imshow(one_img, cmap=cmap)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)


def show_img_and_recon(output, dataset_path, algo_name='vae', epoch_id=None,
                       cmap=None):
    if cmap is None:
        cmap = CMAPS_DICT[algo_name]
    print('Loading %s' % dataset_path)
    dataset = np.load(dataset_path)
    print('Dataset shape:', dataset.shape)

    nrows = 2
    ncols = min(5, len(dataset))
    img = dataset[:ncols]

    encoder = load_module(
        output, algo_name=algo_name, module_name='encoder', epoch_id=epoch_id)
    decoder = load_module(
        output, algo_name=algo_name, module_name='decoder', epoch_id=epoch_id)

    z, _ = encoder(torch.Tensor(img).to(DEVICE))
    recon, _ = decoder(z)
    recon = recon.cpu().detach().numpy()

    data_dim = functools.reduce(
           (lambda x, y: x * y), encoder.img_shape)

    if recon.shape[-1] == data_dim:
        img_side = int(np.sqrt(data_dim))  # HACK
        recon = recon.reshape(
            (-1,) * len(recon.shape[:-1]) + (img_side, img_side))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 8))
    i = 0
    for one_img, one_recon in zip(img, recon):
        if i > ncols - 1:
            break
        if len(one_img.shape) == 3:
            one_img = one_img[0]  # channels
        if len(one_recon.shape) == 3:
            one_recon = one_recon[0]  # channels
        ax = axes[0, i]
        ax.imshow(one_img, cmap=cmap)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax = axes[1, i]
        ax.imshow(one_recon, cmap=cmap)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        i += 1


def show_samples_from_prior(output, fig, outer, i,
                            algo_name='vae', epoch_id=None,
                            sqrt_n_samples=10, cmap=None):
    if cmap is None:
        cmap = CMAPS_DICT[algo_name]
    n_samples = sqrt_n_samples ** 2

    decoder = load_module(
        output, algo_name=algo_name, module_name='decoder', epoch_id=epoch_id)
    data_dim = functools.reduce(
            (lambda x, y: x * y), decoder.img_shape)

    z_from_prior = nn.sample_from_prior(
        latent_dim=decoder.latent_dim, n_samples=n_samples)
    x_recon, _ = decoder(z_from_prior)
    x_recon = x_recon.cpu().detach().numpy()

    inner = gridspec.GridSpecFromSubplotSpec(
        sqrt_n_samples, sqrt_n_samples,
        subplot_spec=outer[i], wspace=0., hspace=0.)

    for i_recon, one_x_recon in enumerate(x_recon):
        ax = plt.Subplot(fig, inner[i_recon])
        img_side = int(np.sqrt(data_dim))
        one_x_recon = one_x_recon.reshape((img_side, img_side))
        ax.imshow(one_x_recon, cmap=cmap)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        fig.add_subplot(ax)


def plot_losses(output, epoch_id=None):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))

    ax = axes[0]
    ax = plot_convergence(
        ax, output, algo_name='vae', crit_name='total',
        epoch_id=epoch_id)

    ax = axes[1]
    ax = plot_convergence(
        ax, output, algo_name='vae', crit_name='reconstruction',
        epoch_id=epoch_id)

    ax = axes[2]
    ax = plot_convergence(
        ax, output, algo_name='vae', crit_name='regularization',
        epoch_id=epoch_id)

    ax = axes[3]
    ax = plot_convergence(
        ax, output, algo_name='vae', crit_name='discriminator',
        epoch_id=epoch_id)
    ax = plot_convergence(
        ax, output, algo_name='vae', crit_name='generator',
        epoch_id=epoch_id)


def plot_variance_explained(output, dataset_path, epoch_id=None):
    mus = analyze.latent_projection(
        output=output, dataset_path=dataset_path, epoch_id=epoch_id)
    n_pca_components = mus.shape[-1]

    pca, projected_mus = analyze.pca_projection(mus, n_pca_components)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    ax = axes[0]
    ax.plot(
        np.arange(1, n_pca_components+1), pca.explained_variance_ratio_)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('PCA components')
    ax.set_ylabel('Percentage of variance explained')
    ax.set_xticks(np.arange(1, n_pca_components+1, step=1))

    ax = axes[1]
    ax.plot(
        np.arange(1, n_pca_components+1),
        np.cumsum(pca.explained_variance_ratio_))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('PCA components')
    ax.set_ylabel('Cumulative sum of variance explained')
    ax.set_xticks(np.arange(1, n_pca_components+1, step=1))
    return ax


def plot_fmri(ax, projected_mus, labels,
              marker_label='task', title='Trajectories', ses_ids=None):
    # ax.plot(projected_mus[:, 0], projected_mus[:, 1], label=title)

    for mu, time, ses, task in zip(
            projected_mus, labels['time'],
            labels['ses'], labels['task']):
        if (ses_ids is not None) and (ses not in ses_ids):
            continue
        colors = np.array([COLORS[time-1]])[0]
        if marker_label == 'task':
            markers = TASK_TO_MARKER[task]
        else:
            markers = SES_TO_MARKER[ses]
        im = ax.plot(mu[0], mu[1], marker=markers, c=colors)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    return im, ax
