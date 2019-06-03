""" Visualization for toy experiments."""

import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import seaborn as sns
import torch

from scipy.stats import gaussian_kde

import imnn
import toylosses
import toynn

ALPHA = 0.2
BINS = 40

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


ALGO_STRINGS = {
    'vae': 'VAE', 'iwae': 'IWAE', 'vem': 'AVEM'}
CRIT_STRINGS = {
    'neg_elbo': 'Neg ELBO',
    'neg_iwelbo': 'Neg IWELBO',
    'neg_loglikelihood': 'NLL'}
TRAIN_VAL_STRINGS = {'train': 'Train', 'val': 'Valid'}
COLOR_DICT = {'neg_elbo': 'red', 'neg_iwelbo': 'orange'}
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


def plot_data_distribution(ax, output_dir, algo_name='vae'):
    n_samples = 1000

    string_base = '%s/train_%s/models/decoder.pth' % (output_dir, algo_name)
    decoder_path = glob.glob(string_base)[0]
    decoder = torch.load(decoder_path, map_location=DEVICE)

    string_base = '%s/synthetic/decoder_true.pth' % output_dir
    decoder_true_path = glob.glob(string_base)[0]
    decoder_true = torch.load(decoder_true_path, map_location=DEVICE)

    generated_true_x = toynn.generate_from_decoder(decoder_true, n_samples)
    generated_x = toynn.generate_from_decoder(decoder, n_samples)

    plot_data(generated_true_x, color='darkgreen', ax=ax)
    plot_data(generated_x, color=ALGO_COLOR_DICT[algo_name], ax=ax)
    ax.set_title('Data distributions p(x)')
    ax.set_xlabel('x')
    return ax


def plot_posterior(ax, output_dir, algo_name='vae'):
    n_to_sample = 10000
    w_true = 2
    x = 3

    string_base = '%s/train_%s/models/encoder.pth' % (output_dir, algo_name)
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


def print_average_time(output_dir, algo_name='vae'):
    string_base = '%s/train_%s/val_losses.pkl' % (output_dir, algo_name)
    losses_path = glob.glob(string_base)[0]
    losses_all_epochs = pickle.load(open(losses_path, 'rb'))

    times = [loss['total_time'] for loss in losses_all_epochs]
    print(np.mean(times))


def print_weights(output_dir, algo_name='vae', module_name='decoder'):
    string_base = '%s/train_%s/models/%s.pth' % (
        output_dir, algo_name, module_name)
    module_path = glob.glob(string_base)[0]
    module = torch.load(module_path, map_location=DEVICE)

    print('\n-- Learnt values of parameters for module %s' % module_name)
    for name, param in module.named_parameters():
        print(name, param.data, '\n')


def plot_weights(ax, output_dir, algo_name='vae',
                 from_epoch=0, to_epoch=1000, color='blue', dashes=False):
    string_base = '%s/train_%s/train_losses.pkl' % (output_dir, algo_name)
    losses_path = glob.glob(string_base)[0]
    losses_all_epochs = pickle.load(open(losses_path, 'rb'))

    weight_w = [loss['weight_w'] for loss in losses_all_epochs]
    weight_phi = [loss['weight_phi'] for loss in losses_all_epochs]

    # Take absolute value to avoid identifiability problem
    weight_w = np.abs(weight_w)
    weight_phi = np.abs(weight_phi)

    n_epochs = len(weight_w)
    to_epoch = min(to_epoch, n_epochs)

    label = '%s' % ALGO_STRINGS[algo_name]

    if not dashes:
        ax.plot(weight_w[from_epoch:to_epoch], weight_phi[from_epoch:to_epoch],
                label=label, color=color)
    else:
        ax.plot(weight_w[from_epoch:to_epoch], weight_phi[from_epoch:to_epoch],
                label=label, color=color, dashes=[2, 2, 2, 2])

    return ax


def get_losses(output_dir, algo_name='vae',
               crit_name='neg_elbo', mode='train'):
    string_base = '%s/train_%s/%s_losses.pkl' % (output_dir, algo_name, mode)
    losses_path = glob.glob(string_base)[0]
    if len(losses_path) == 0:
        string_base = '%s/train_%s/models/epoch_*_checkpoint.pth' % (
            output_dir, algo_name)
        ckpts = glob.glob(string_base)
        if len(ckpts) == 0:
            msg = '%s: No %s_losses.pkl found. No checkpoints found.' % (
                ALGO_STRINGS[algo_name], mode)
            print(msg)
        else:
            ckpts_ids_and_paths = [(int(f.split('_')[2]), f) for f in ckpts]
            ckpt_id, ckpt_path = max(
                ckpts_ids_and_paths, key=lambda item: item[0])
            print('Found checkpoints. Getting: %s.' % ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)

            losses = ckpt['%s_losses' % mode]
            losses = [loss[crit_name] for loss in losses]

    else:
        losses = pickle.load(open(losses_path, 'rb'))
        losses = [loss[crit_name] for loss in losses]
    return losses


def plot_criterion(ax, output_dir,
                   algo_name='vae', crit_name='neg_elbo', mode='train',
                   from_epoch=0, to_epoch=1000, color='blue', dashes=False):
    string_base = '%s/train_%s/%s_losses.pkl' % (output_dir, algo_name, mode)
    losses_path = glob.glob(string_base)[0]
    losses_all_epochs = pickle.load(open(losses_path, 'rb'))

    losses_total = [loss[crit_name] for loss in losses_all_epochs]

    n_epochs = len(losses_total)
    epochs = range(n_epochs)
    to_epoch = min(to_epoch, n_epochs)

    label = '%s: %s %s' % (
        ALGO_STRINGS[algo_name],
        TRAIN_VAL_STRINGS[mode],
        CRIT_STRINGS[crit_name])

    if not dashes:
        ax.plot(epochs[from_epoch:to_epoch], losses_total[from_epoch:to_epoch],
                label=label, color=color)
    else:
        ax.plot(epochs[from_epoch:to_epoch], losses_total[from_epoch:to_epoch],
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


def plot_kl_posterior(ax, output_dir, algo_name='vae', mode='train', n_train=0,
                      from_epoch=0, to_epoch=1000, color='blue', dashes=False):

    string_base = '%s/train_%s/%s_losses.pkl' % (output_dir, algo_name, mode)
    losses_path = glob.glob(string_base)[0]
    losses_all_epochs = pickle.load(open(losses_path, 'rb'))

    weight_w = [loss['weight_w'] for loss in losses_all_epochs]
    weight_phi = [loss['weight_phi'] for loss in losses_all_epochs]

    # Take absolute value to avoid identifiability problem
    weight_w = np.abs(weight_w)
    weight_phi = np.abs(weight_phi)

    n_epochs = len(weight_w)
    epochs = range(n_epochs)
    to_epoch = min(to_epoch, n_epochs)

    string_base = '%s/synthetic/dataset.npy' % output_dir
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
        ax.plot(epochs[from_epoch:to_epoch], kl[from_epoch:to_epoch],
                label=label, color=color)
    else:
        ax.plot(epochs[from_epoch:to_epoch], kl[from_epoch:to_epoch],
                label=label, color=color, dashes=[2, 2, 2, 2])
    ax.set_xlabel('epochs')
    return ax


def plot_kl_posterior_bis(ax, output_dir, algo_name='vae', mode='train',
                          from_epoch=0, to_epoch=1000, color='blue',
                          dashes=False):

    neg_elbo = get_losses(
        output_dir=output_dir, algo_name=algo_name,
        crit_name='neg_elbo', mode=mode)
    neg_ll = get_losses(
        output_dir=output_dir, algo_name=algo_name,
        crit_name='neg_loglikelihood', mode=mode)

    kl = [nelbo - nll for (nelbo, nll) in zip(neg_elbo, neg_ll)]

    n_epochs = len(kl)
    epochs = range(n_epochs)
    to_epoch = min(to_epoch, n_epochs)

    label = '%s: %s KL' % (
        ALGO_STRINGS[algo_name], TRAIN_VAL_STRINGS[mode])

    if not dashes:
        ax.plot(epochs[from_epoch:to_epoch], kl[from_epoch:to_epoch],
                label=label, color=color)
    else:
        ax.plot(epochs[from_epoch:to_epoch], kl[from_epoch:to_epoch],
                label=label, color=color, dashes=[2, 2, 2, 2])
    ax.set_xlabel('epochs')
    return ax


def plot_convergence(ax, output_dir, algo_name, crit_name,
                     from_epoch=0, to_epoch=1000):
    ax = plot_criterion(
        ax, output_dir, algo_name=algo_name, crit_name=crit_name, mode='train',
        from_epoch=from_epoch, to_epoch=to_epoch, color=COLOR_DICT[crit_name])
    ax = plot_criterion(
        ax, output_dir, algo_name=algo_name, crit_name=crit_name, mode='val',
        from_epoch=from_epoch, to_epoch=to_epoch, color=COLOR_DICT[crit_name],
        dashes=True)
    ax.set_title('Convergence of %s.' % ALGO_STRINGS[algo_name])
    return ax


def min_neg_ll(output_dir, val=False):
    dataset_path = glob.glob(
        '%s/synthetic/dataset.npy' % output_dir)[0]
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


def elbo_neg_ll(output_dir, val='False'):
    dataset_path = glob.glob(
        '%s/synthetic/dataset.npy' % output_dir)[0]
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


def print_optimums(output_dir):
    train_w_mle, train_min_neg_ll = min_neg_ll(output_dir)
    val_w_mle, val_min_neg_ll = min_neg_ll(output_dir, val=True)

    train_w_elbo, train_elbo_neg_ll = elbo_neg_ll(output_dir)
    val_w_elbo, val_elbo_neg_ll = elbo_neg_ll(output_dir, val=True)

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


def plot_weights_landscape(ax, output_dir, from_epoch=0, to_epoch=1000):
    ax = plot_weights(
        ax, output_dir, algo_name='vae',
        from_epoch=from_epoch, to_epoch=to_epoch, color='red')
    ax = plot_weights(
        ax, output_dir, algo_name='iwae',
        from_epoch=from_epoch, to_epoch=to_epoch, color='orange')
    ax = plot_weights(
        ax, output_dir, algo_name='vem',
        from_epoch=from_epoch, to_epoch=to_epoch, color='blue')

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


def plot_nll(ax, output_dir, algo_name, from_epoch=0, to_epoch=1000):
    ax = plot_criterion(
        ax, output_dir, algo_name=algo_name, crit_name='neg_loglikelihood',
        mode='train', from_epoch=from_epoch, to_epoch=to_epoch,
        color=ALGO_COLOR_DICT[algo_name])
    ax = plot_criterion(
        ax, output_dir, algo_name=algo_name, crit_name='neg_loglikelihood',
        mode='val', from_epoch=from_epoch, to_epoch=to_epoch,
        color=ALGO_COLOR_DICT[algo_name], dashes=True)
    return ax


def plot_kl(ax, output_dir, algo_name, from_epoch=0, to_epoch=1000):
    ax = plot_kl_posterior(
        ax, output_dir, algo_name=algo_name, mode='train',
        from_epoch=from_epoch, to_epoch=to_epoch,
        color=ALGO_COLOR_DICT[algo_name], dashes=False)
    ax = plot_kl_posterior(
        ax, output_dir, algo_name=algo_name, mode='val',
        from_epoch=from_epoch, to_epoch=to_epoch,
        color=ALGO_COLOR_DICT[algo_name], dashes=True)

    return ax


def load_module(output_dir, algo_name='vae', module_name='encoder',
                latent_dim=20, data_dim=784):
    # TODO(nina): Delete the need of knowing the VAE's architecture
    # in order to load it
    module_path = glob.glob(
        '%s/train_%s/models/%s.pth' % (
             output_dir, algo_name, module_name))
    if len(module_path) == 0:
        ckpts = glob.glob(
            '%s/train_%s/models/epoch_*_checkpoint.pth' % (
                output_dir, algo_name))
        if len(ckpts) == 0:
            print('No module found. No checkpoints found.')
        else:
            ckpts_ids_and_paths = [(int(f.split('_')[2]), f) for f in ckpts]
            ckpt_id, ckpt_path = max(
                ckpts_ids_and_paths, key=lambda item: item[0])
            print('Found checkpoints. Getting: %s.' % ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)

            vae = imnn.VAE(
                latent_dim=latent_dim,
                data_dim=data_dim)
            vae.to(DEVICE)

            modules = {}
            modules['encoder'] = vae.encoder
            modules['decoder'] = vae.decoder
            module = modules[module_name]
            module_ckpt = ckpt[module_name]
            module.load_state_dict(module_ckpt['module_state_dict'])

    else:
        module_path = module_path[0]
        print('Loading: %s' % module_path)
        module = torch.load(module_path, map_location=DEVICE)
    return module


def show_samples(output_dir, fig, outer, i, algo_name='vae',
                 latent_dim=20, data_dim=784, sqrt_n_samples=10,
                 cmap=None):
    if cmap is None:
        cmap = CMAPS_DICT[algo_name]
    n_samples = sqrt_n_samples ** 2

    decoder = load_module(
        output_dir, algo_name=algo_name, module_name='decoder',
        latent_dim=latent_dim, data_dim=data_dim)

    z_from_prior = imnn.sample_from_prior(
        latent_dim=latent_dim, n_samples=n_samples)
    x_recon, _ = decoder(z_from_prior)
    x_recon = x_recon.cpu().detach().numpy()

    inner = gridspec.GridSpecFromSubplotSpec(
        sqrt_n_samples, sqrt_n_samples,
        subplot_spec=outer[i], wspace=0., hspace=0.)

    for i_recon, one_x_recon in enumerate(x_recon):
        ax = plt.Subplot(fig, inner[i_recon])
        one_x_recon = one_x_recon.reshape((28, 28))
        ax.imshow(one_x_recon, cmap=cmap)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        fig.add_subplot(ax)
