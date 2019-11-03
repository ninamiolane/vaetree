"""Visualization for experiments."""

import functools
import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import seaborn as sns
import torch

import geomstats.visualization as visualization
from matplotlib import animation
from scipy.stats import gaussian_kde
# from torch_viz import make_dot

import analyze
import nn
import toylosses
import toynn
import train_utils


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
    'vae': 'VAE', 'iwae': 'IWAE', 'vem': 'AVEM', 'vem_02': 'AVEM 20%'}
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
    'neg_loglikelihood': 'blue',
    'neg_elbo': 'red',
    'neg_iwelbo': 'orange',
    'total': 'lightblue',
    'reconstruction': 'lightgreen',
    'regularization': 'darkred',
    'discriminator': 'purple',
    'generator': 'violet'}
ALGO_COLOR_DICT = {
    'vae': 'red',
    'iwae': 'orange',
    'vem': 'blue',
    'vem_02': 'darkblue'}
CMAPS_DICT = {'vae': 'Reds', 'iwae': 'Oranges', 'vem': 'Blues'}
MANIFOLD_VIS_DICT = {'s2': 'S2', 'h2': 'H2_poincare_disk'}


VAE_TYPE_COLOR_DICT = {
    'vae': 'C0',
    'vae_proj': 'C1',
    'pga': 'C3',
    'gvae': 'C4',
    'gvae_tgt': 'darkgreen'
}

N_MARKERS_DICT = {
    '10k': 's',
    '100k': 'o'
}


FRAC_VAL = 0.2
N_SAMPLES = 10000
N_TRAIN = int((1 - FRAC_VAL) * N_SAMPLES)

FOCUS_MAX = 2.
start = 0.5
by = 0.5
num = int((FOCUS_MAX - start) / by + 1)
colormap = cm.get_cmap('viridis')
COLORS_FOCUS = colormap(np.linspace(start=0, stop=1, num=num))

start = -180
by = 1
num = 2 * 180 + 1
colormap = cm.get_cmap('twilight')
COLORS_THETA = colormap(np.linspace(start=0, stop=1, num=num))

COLORS = {
    'focus': COLORS_FOCUS,
    'theta': COLORS_THETA
}


def plot_data(x_data, color='darkgreen', label=None, s=20, alpha=0.3, ax=None):
    print('inplotdata')
    _, data_dim = x_data.shape
    if data_dim == 1:
        ax.hist(
            x_data, bins=BINS, alpha=ALPHA,
            color=color, label=label, density=True)
    elif data_dim == 2:
        sns.scatterplot(
            x_data[:, 0], x_data[:, 1], x_data[:, 2],
            ax=ax, label=label, color=color, alpha=alpha, s=s)
    else:
        sns.scatterplot(
            x_data[:, 0], x_data[:, 1],
            ax=ax, label=label, color=color, alpha=alpha, s=s)
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


def load_losses(output, epoch_id=None,
                crit_name='neg_elbo', mode='train'):
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)

    losses = ckpt['%s_losses' % mode]
    losses = [loss[crit_name] for loss in losses]

    return losses


def plot_criterion(ax, output, crit_name='neg_elbo',
                   mode='train', start_epoch_id=0, epoch_id=None,
                   color='blue', dashes=False):

    losses_total = load_losses(
        output=output,
        crit_name=crit_name, mode=mode, epoch_id=epoch_id)

    n_epochs = len(losses_total)
    epochs = range(n_epochs)
    if epoch_id is not None:
        epoch_id = min(epoch_id, n_epochs)

    label = '%s Loss' % (
        #ALGO_STRINGS[algo_name],
        TRAIN_VAL_STRINGS[mode])
        #CRIT_STRINGS[crit_name])
    #label = '%s: %s %s' % (
    #    ALGO_STRINGS[algo_name],
    #    TRAIN_VAL_STRINGS[mode],
    #    CRIT_STRINGS[crit_name])

    if not dashes:
        ax.plot(
            epochs[start_epoch_id:epoch_id],
            losses_total[start_epoch_id:epoch_id],
            label=label, color=color, linewidth=3)
    else:
        ax.plot(
            epochs[start_epoch_id:epoch_id],
            losses_total[start_epoch_id:epoch_id],
            label=label, color=color, dashes=[2, 2, 2, 2], linewidth=3)
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


def plot_convergence(ax, output, crit_name,
                     start_epoch_id=0, epoch_id=None):
    ax = plot_criterion(
        ax, output, crit_name=crit_name, mode='train',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id,
        color=COLOR_DICT[crit_name])
    ax = plot_criterion(
        ax, output, crit_name=crit_name, mode='val',
        start_epoch_id=start_epoch_id, epoch_id=epoch_id,
        color=COLOR_DICT[crit_name], dashes=True)
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
    np.random.shuffle(dataset)

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
    plt.tight_layout()


def get_recon(output, img, algo_name='vae', epoch_id=None,
              cmap=None):

    encoder = train_utils.load_module(
        output, module_name='encoder', epoch_id=epoch_id)
    decoder = train_utils.load_module(
        output, module_name='decoder', epoch_id=epoch_id)
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)

    if 'spd_feature' in ckpt['nn_architecture']:
        spd_feature = ckpt['nn_architecture']['spd_feature']

        if spd_feature is not None:
            img = train_utils.spd_feature_from_matrix(
                img, spd_feature=spd_feature)
    z, _ = encoder(torch.Tensor(img).to(DEVICE))
    recon, _ = decoder(z)
    recon = recon.cpu().detach().numpy()
    z = z.cpu().detach().numpy()

    try:
        data_dim = functools.reduce(
               (lambda x, y: x * y), encoder.img_shape)
    except AttributeError:
        data_dim = encoder.data_dim

    if 'spd_feature' in ckpt['nn_architecture']:
        if spd_feature is not None:
            recon = train_utils.matrix_from_spd_feature(
                recon, spd_feature=spd_feature)

    if recon.shape[-1] == data_dim:
        img_side = int(np.sqrt(data_dim))  # HACK
        recon = recon.reshape(
            (-1,) * len(recon.shape[:-1]) + (img_side, img_side))

    return recon


def show_img_and_recon(output, dataset_path, algo_name='vae', epoch_id=None,
                       cmap=None):
    if cmap is None:
        cmap = CMAPS_DICT[algo_name]

    print('Loading %s' % dataset_path)
    dataset = np.load(dataset_path)
    print('Dataset shape:', dataset.shape)

    nrows = 2
    ncols = min(5, len(dataset))
    img = dataset[10:ncols+10]

    recon = get_recon(
        output=output, img=img, algo_name=algo_name, epoch_id=epoch_id)

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
        if i != 0:
            assert np.all(recon[i] != recon[i-1])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        i += 1


def recon_from_z(output, z, algo_name='vae', epoch_id=None):
    decoder = train_utils.load_module(
        output, algo_name=algo_name, module_name='decoder', epoch_id=epoch_id)
    recon, _ = decoder(z)
    recon = recon.cpu().detach().numpy()
    return recon


def show_samples_from_prior(output, fig, outer, i,
                            algo_name='vae', epoch_id=None,
                            sqrt_n_samples=10, cmap=None):
    if cmap is None:
        cmap = CMAPS_DICT[algo_name]
    n_samples = sqrt_n_samples ** 2

    decoder = train_utils.load_module(
        output, algo_name=algo_name, module_name='decoder', epoch_id=epoch_id)

    z_from_prior = nn.sample_from_prior(
        latent_dim=decoder.latent_dim, n_samples=n_samples)
    x_recon, _ = decoder(z_from_prior)
    x_recon = x_recon.cpu().detach().numpy()
    ckpt = train_utils.load_checkpoint(
        output=output, epoch_id=epoch_id)
    spd_feature = ckpt['nn_architecture']['spd_feature']
    if spd_feature is not None:
        x_recon = train_utils.matrix_from_spd_feature(
            x_recon, spd_feature=spd_feature)
        # Assume x_reocn.shape = n_data, 15, 15 no channels
        x_recon = x_recon.reshape((-1, x_recon.shape[1]*x_recon.shape[2]))

    inner = gridspec.GridSpecFromSubplotSpec(
        sqrt_n_samples, sqrt_n_samples,
        subplot_spec=outer[i], wspace=0., hspace=0.)

    img_side = int(np.sqrt(x_recon.shape[1]))
    for i_recon, one_x_recon in enumerate(x_recon):
        ax = plt.Subplot(fig, inner[i_recon])
        one_x_recon = one_x_recon.reshape((img_side, img_side))
        ax.imshow(one_x_recon, cmap=cmap)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        fig.add_subplot(ax)


def plot_losses(output, epoch_id=None):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))

    ax = axes[0]
    ax = plot_convergence(
        ax, output, crit_name='total',
        epoch_id=epoch_id)

    ax = axes[1]
    ax = plot_convergence(
        ax, output, crit_name='reconstruction',
        epoch_id=epoch_id)

    ax = axes[2]
    ax = plot_convergence(
        ax, output, crit_name='regularization',
        epoch_id=epoch_id)

    ax = axes[3]
    ax = plot_convergence(
        ax, output, crit_name='discriminator',
        epoch_id=epoch_id)
    ax = plot_convergence(
        ax, output, crit_name='generator',
        epoch_id=epoch_id)


def plot_variance_explained(output, dataset_path, epoch_id=None, axes=None):
    mus = analyze.latent_projection(
        output=output, dataset_path=dataset_path, epoch_id=epoch_id)
    n_pca_components = mus.shape[-1]

    pca, projected_mus = analyze.pca_projection(mus, n_pca_components)

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    ax = axes[0]
    ax.plot(
        np.arange(1, n_pca_components+1), pca.explained_variance_ratio_,
        label='Latent dim: %d' % n_pca_components)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('PCA components')
    ax.set_ylabel('Percentage of variance explained')
    ax.set_xticks(np.arange(1, n_pca_components+1, step=1))

    ax = axes[1]
    ax.plot(
        np.arange(1, n_pca_components+1),
        np.cumsum(pca.explained_variance_ratio_),
        label='Latent dim: %d' % n_pca_components)
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


def plot_nn_graph(module_name='encoder', epoch_id=None):
    module = train_utils.load_module(
        module_name=module_name, epoch_id=epoch_id)
    if module_name == 'encoder':
        in_shape = module.img_shape
    elif module_name == 'decoder':
        in_shape = module.in_shape
    else:
        in_shape = (1, 1, 1, 1)  # placeholder
    x = torch.zeros(
        in_shape[0], in_shape[1], in_shape[2], in_shape[3],
        dtype=torch.float, requires_grad=False)
    out = module(x)
    # make_dot(out)


def generate_submanifolds(output, algo_name, epoch,
                          n_samples=1000,
                          manifold_name='r2',
                          tangent_space=False):
    decoder_true_path = glob.glob('%s/synthetic/decoder_true.pth' % output)[0]
    decoder_true = torch.load(decoder_true_path, map_location=DEVICE)
    decoder = train_utils.load_module(
        output, algo_name, module_name='decoder', epoch_id=int(epoch))

    ckpt = train_utils.load_checkpoint(
        output, epoch_id=epoch)
    logvarx_true = ckpt['nn_architecture']['logvarx_true']

    if manifold_name == 'r2':
        true_x = toynn.generate_from_decoder_fixed_var(
            decoder_true, logvarx=logvarx_true, n_samples=n_samples)
        _, true_x_novarx, _ = decoder_true.generate(n_samples)
        true_x_novarx = true_x_novarx.detach().cpu().numpy()

        x = toynn.generate_from_decoder(decoder, n_samples)
        _, x_novarx, _ = decoder.generate(n_samples)
        x_novarx = x_novarx.detach().cpu().numpy()
    elif manifold_name == 's2' or manifold_name == 'h2':
        if tangent_space is True:
            true_x_novarx = toynn.generate_from_decoder_fixed_var_tgt(
                decoder_true, logvarx=-1000, n_samples=n_samples,
                manifold_name=manifold_name)

            true_x = toynn.generate_from_decoder_fixed_var_tgt(
                decoder_true, logvarx=logvarx_true, n_samples=n_samples,
                manifold_name=manifold_name)

            x_novarx = toynn.generate_from_decoder_fixed_var_tgt(
                decoder, logvarx=-1000, n_samples=n_samples,
                manifold_name=manifold_name)
            # TODO(nina): here logvar is 0 because of the
            # default setting in training: adapt?
            x = toynn.generate_from_decoder_fixed_var_tgt(
                decoder, logvarx=-5, n_samples=n_samples,
                manifold_name=manifold_name)

        else:
            true_x_novarx = toynn.generate_from_decoder_fixed_var_riem(
                decoder_true, logvarx=-1000, n_samples=n_samples,
                manifold_name=manifold_name)

            true_x = toynn.generate_from_decoder_fixed_var_riem(
                decoder_true, logvarx=logvarx_true, n_samples=n_samples,
                manifold_name=manifold_name)

            x_novarx = toynn.generate_from_decoder_fixed_var_riem(
                decoder, logvarx=-1000, n_samples=n_samples,
                manifold_name=manifold_name)

            # TODO(nina): here logvar is 0 because of the
            # default setting in training: adapt?
            x = toynn.generate_from_decoder_fixed_var_riem(
                decoder, logvarx=-5, n_samples=n_samples,
                manifold_name=manifold_name)
    else:
        raise ValueError('Manifold not supported.')


def get_ax_id(ncols, row_id, col_id):
    ax_id = row_id * ncols + col_id + 1
    return ax_id


def get_ax(fig, nrows, ncols, row_id, col_id,
           manifold_name='r2'):
    if manifold_name == 's2':
        ax = fig.add_subplot(
            nrows, ncols,
            get_ax_id(ncols, row_id=row_id, col_id=col_id),
            projection='3d')
    else:
        ax = fig.add_subplot(
            nrows, ncols,
            get_ax_id(ncols, row_id=row_id, col_id=col_id))
    return ax


def plot_true_submanifold(fig, nrows, ncols, row_id, col_id,
                          output, n_samples,
                          algo_name='vae', manifold_name='r2',
                          epoch_id=None, with_noise=False, ax=None,
                          label=''):
    t = np.random.normal(size=(n_samples,))

    true_x_novarx, true_x = analyze.true_submanifold_from_t_and_output(
        t=t, output=output,
        algo_name=algo_name, manifold_name=manifold_name,
        epoch_id=epoch_id, with_noise=with_noise)

    if ax is None:
        ax = get_ax(
                fig, nrows, ncols, row_id, col_id,
                manifold_name)

    if manifold_name == 'r2':
        _ = ax.scatter(
                true_x_novarx[:, 0], true_x_novarx[:, 1],
                color='lime', alpha=1, s=30,
                label=label)
        if with_noise:
            _ = ax.scatter(
                    true_x[:, 0], true_x[:, 1],
                    color='green', alpha=1, s=30,
                    label='Data')

    elif manifold_name in ['s2', 'h2']:
        ax = visualization.plot(
                true_x_novarx, ax=ax,
                space=MANIFOLD_VIS_DICT[manifold_name],
                color='lime', alpha=1, s=20,
                label=label)
        if with_noise:
            ax = visualization.plot(
                    true_x, ax=ax,
                    space=MANIFOLD_VIS_DICT[manifold_name],
                    color='green', alpha=1, s=20,
                    label='Data')

    if manifold_name != 'r2':
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if manifold_name == 's2':
        ax.set_zticks([])

    if manifold_name == 'h2':
        ax.set_xlabel('')
        ax.set_ylabel('')
    return ax


def plot_learned_submanifold(fig, nrows, ncols, row_id, col_id,
                             logvarx_true, n, n_samples, algo_name='vae',
                             manifold_name='r2', vae_type='gvae_tgt',
                             epoch_id=None, with_noise=False,
                             ax=None,
                             color='black', s=20, label=''):
    t = np.random.normal(size=(n_samples,))
    x_novarx, x = analyze.learned_submanifold_from_t_and_vae_type(
        t, vae_type, logvarx_true, n,
        algo_name=algo_name, manifold_name=manifold_name,
        epoch_id=epoch_id, with_noise=with_noise)

    if ax is None:
        ax = get_ax(
                fig, nrows, ncols, row_id, col_id,
                manifold_name)

    if manifold_name == 'r2':
        _ = ax.scatter(
                x_novarx[:, 0], x_novarx[:, 1],
                color=color, alpha=1, s=s,
                label='Learned weighted submanifold')
        if with_noise:
            _ = ax.scatter(
                    x[:, 0], x[:, 1],
                    color=color, alpha=1, s=s,
                    label='Data')

    elif manifold_name in ['s2', 'h2'] and vae_type != 'vae':
        ax = visualization.plot(
                x_novarx, ax=ax,
                space=MANIFOLD_VIS_DICT[manifold_name],
                color=color, alpha=1, s=s,
                label='Learned weighted submanifold')
        if with_noise:
            ax = visualization.plot(
                    x, ax=ax,
                    space=MANIFOLD_VIS_DICT[manifold_name],
                    color=color, alpha=1, s=s,
                    label='Data')

    elif manifold_name == 'r3' or vae_type == 'vae':
        _ = ax.scatter(
            x_novarx[:, 0], x_novarx[:, 1], x_novarx[:, 2],
            color=color, alpha=1, s=s,
            label=label)

        if with_noise:
            _ = ax.scatter(
                x[:, 0], x[:, 1], x[:, 2],
                color=color, alpha=1, s=s,
                label=label)

    else:
        ValueError('Manifold not supported.')

    if manifold_name != 'r2':
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if manifold_name == 's2':
        ax.set_zticks([])

    if manifold_name == 'h2':
        ax.set_xlabel('')
        ax.set_ylabel('')

    return ax


def plot_submanifolds(ax, epoch,
                      generated_true_x, generated_true_x_novarx,
                      generated_x, generated_x_novarx, algo_name,
                      xlim=(-15, 15), ylim=(-45, 5),
                      manifold_name='r2', tangent_space=False):
    if manifold_name == 'r2' or tangent_space is True:
        xlim = (-0.8, 0.8)
        ylim = (-1.5, 0.5)
        ax = plot_data(
            generated_true_x_novarx, color='lime',
            label='True submanifold', ax=ax)
        ax = plot_data(
            generated_true_x, color='green',
            label='Samples from true generator', ax=ax)
        ax = plot_data(
            generated_x_novarx, color='black',
            label='Learned submanifold', ax=ax)
        ax = plot_data(
            generated_x, color=ALGO_COLOR_DICT[algo_name],
            label='Samples from learned generator', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title('%s at epoch %d' % (ALGO_STRINGS[algo_name], epoch))
        ax.grid(True)
    else:
        manifold_vis = MANIFOLD_VIS_DICT[manifold_name]
        visualization.plot(
            generated_x, ax=ax, space=manifold_vis,
            color=ALGO_COLOR_DICT[algo_name], alpha=0.3)
        if manifold_name == 'h2':
            visualization.plot(
                generated_x_novarx, ax=ax, space=manifold_vis,
                color='black', alpha=0.3)
            visualization.plot(
                generated_true_x, ax=ax, space=manifold_vis,
                color='green', alpha=0.3)
            visualization.plot(
                generated_true_x_novarx, ax=ax, space=manifold_vis,
                color='lime', alpha=0.1)
        else:
            ax.scatter(
                    generated_x_novarx[:, 0],
                    generated_x_novarx[:, 1],
                    generated_x_novarx[:, 2], color='black', alpha=0.3)
            ax.scatter(
                    generated_true_x[:, 0],
                    generated_true_x[:, 1],
                    generated_true_x[:, 2], color='green', alpha=0.3)
            ax.scatter(
                    generated_true_x_novarx[:, 0],
                    generated_true_x_novarx[:, 1],
                    generated_true_x_novarx[:, 2], color='lime', alpha=0.1)
    return ax


def get_unexplained_variance(output, dataset_path, variance_name='eucl'):
    """
    For variance_name == eucl
    Use L2 norm between data points to compute unexplained variance:
        unexplained_variance
            = sum_i (img_i - recon_i)**2 / sum_i (img_i - mean) ** 2
    It amounts to the residual variance in the Euclidean space with L2 norm.

    For variance_name == log_eucl
    For SSD matrices, compute residual variances on the log-matrices.
    """
    print('Loading %s' % dataset_path)
    img = np.load(dataset_path)
    print('Dataset shape:', img.shape)

    recon = get_recon(output, img, algo_name='vae')

    if variance_name == 'log_eucl':
        img = train_utils.spd_feature_from_matrix(img, 'log_matrix')
        recon = np.expand_dims(recon, axis=1)
        recon = train_utils.spd_feature_from_matrix(recon, 'log_matrix')

    img = np.squeeze(img)
    recon = np.squeeze(recon)
    assert len(recon.shape) == 3
    assert len(img.shape) == 3

    ssd = np.sum((img - recon)**2, axis=(1, 2))
    mean_ssd = np.mean(ssd)

    mean_img = np.mean(img, axis=0)

    variance = np.mean(np.sum((img - mean_img)**2, axis=(1, 2)))

    unexplained_var = mean_ssd / variance
    return unexplained_var


def plot_cryo(ax, output, img_path, labels_path,
              n_pc=2, label_name='focus', epoch_id=None):
    projected_mus, labels = analyze.get_cryo(
        output, img_path, labels_path, n_pca_components=n_pc, epoch_id=epoch_id)
    colored_labels = labels[label_name]
    if label_name == 'focus':
        colored_labels = [focus / 10000. for focus in colored_labels]

    for mu, colored_label in zip(projected_mus, colored_labels):
        #if label_name == 'theta' and focus != 2.5:
        #    continue
        if label_name == 'focus':##
            color_id = int(2 * colored_label) - 1
            if color_id > 3:
                color_id = 3

        elif label_name == 'theta':
            color_id = int((colored_label + 180))

        colors = COLORS[label_name]
        if n_pc == 2:
            im = ax.scatter(mu[0], mu[1], c=np.array([colors[color_id]]), s=20)
        else:
            im = ax.scatter(mu[0], mu[1], mu[2], c=np.array([colors[color_id]]))
    return im, ax


def hist_labels(labels):
    fig = plt.figure(figsize=(24, 16))

    ax = fig.add_subplot(411)
    ax = ax.hist(labels['focus'], bins=20)

    ax = fig.add_subplot(412)
    ax = ax.hist(labels['theta'], bins=45)

    ax = fig.add_subplot(413)
    ax = ax.hist(labels['theta'], bins=90)

    ax = fig.add_subplot(414)
    ax = ax.hist(labels['theta'], bins=180)
