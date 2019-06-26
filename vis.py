"""Visualization for experiments."""

import glob
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os
import pickle
import torch

import nn

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


def load_last_module(output_dir, module_name):
    models = glob.glob(
        '%s/training/models/*_%s_*.pth' % (output_dir, module_name))

    model_ids = [(int(f.split('_')[-8]), f) for f in models]

    start_epoch, last_cp = max(model_ids, key=lambda item: item[0])
    print('Last checkpoint: ', last_cp)
    model = torch.load(last_cp, map_location=DEVICE)
    return model


def load_module(output_dir, module_name, epoch_id):
    model = glob.glob(
        '%s/training/models/epoch_%d_%s_*.pth' % (
            output_dir, epoch_id, module_name))[0]
    print('Loading: %s' % model)
    model = torch.load(model, map_location=DEVICE)
    return model


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


def show_samples_from_prior(output_dir, epoch_id, sqrt_n_samples=3,
                            latent_dim=None, cmap='gray'):
    n_samples = sqrt_n_samples ** 2

    decoder = load_module(
        output_dir, module_name='decoder', epoch_id=epoch_id)

    z_from_prior = nn.sample_from_prior(
        latent_dim=latent_dim, n_samples=n_samples)
    x_recon, _ = decoder(z_from_prior)
    x_recon = x_recon.cpu().detach().numpy()

    fig, axes = plt.subplots(
        nrows=sqrt_n_samples, ncols=sqrt_n_samples, figsize=(6, 6))

    for i_recon, one_x_recon in enumerate(x_recon):
        one_x_recon = one_x_recon[0]  # channels
        ax = axes[i_recon % sqrt_n_samples, int(i_recon // sqrt_n_samples)]
        ax.imshow(one_x_recon, cmap=cmap)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)


def plot_losses(output_dir, epoch_id):

    loss_types = [
        'total',
        'discriminator', 'generator',
        'reconstruction', 'regularization']
    train_losses = {loss_type: [] for loss_type in loss_types}
    val_losses = {loss_type: [] for loss_type in loss_types}

    for i in range(epoch_id+1):
        losses_filename = os.path.join(
            output_dir, 'training/losses/epoch_%d.pkl' % epoch_id)
        train_val = pickle.load(open(losses_filename, 'rb'))
        train = train_val['train']
        val = train_val['val']

        for loss_type in loss_types:
            loss = train[loss_type]
            train_losses[loss_type].append(loss)

            loss = val[loss_type]
            val_losses[loss_type].append(loss)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9))

    # Total
    ax = axes[0, 0]

    ax.plot(train_losses['total'])
    ax.set_title('Train Loss')

    ax = axes[1, 0]
    ax.plot(val_losses['total'])
    ax.set_title('Val Loss')

    # Decomposed in sublosses
    epochs = range(epoch_id+1)

    ax = axes[0, 1]
    ax.plot(epochs, train_losses['discriminator'])
    ax.plot(epochs, train_losses['generator'])
    ax.plot(epochs, train_losses['reconstruction'])
    ax.plot(epochs, train_losses['regularization'])

    ax.set_title('Train Loss Decomposed')
    ax.legend(
        [loss_type for loss_type in loss_types if loss_type != 'total'],
        loc='upper right')

    ax = axes[1, 1]
    ax.plot(epochs, val_losses['discriminator'])
    ax.plot(epochs, val_losses['generator'])
    ax.plot(epochs, val_losses['reconstruction'])
    ax.plot(epochs, val_losses['regularization'])

    ax.set_title('Val Loss Decomposed')
    ax.legend(
        [loss_type for loss_type in loss_types if loss_type != 'total'],
        loc='upper right')

    # Only Discriminator and Generator
    ax = axes[0, 2]
    ax.plot(epochs, train_losses['discriminator'])
    ax.plot(epochs, train_losses['generator'])
    ax.set_title('Train Loss: Discriminator and Generator only')
    ax.legend(
        [loss_type for loss_type in loss_types
         if loss_type == 'discriminator' or loss_type == 'generator'],
        loc='upper right')

    ax = axes[1, 2]
    ax.plot(epochs, val_losses['discriminator'])
    ax.plot(epochs, val_losses['generator'])
    ax.set_title('Val Loss: Discriminator and Generator only')
    ax.legend(
        [loss_type for loss_type in loss_types
         if loss_type == 'discriminator' or loss_type == 'generator'],
        loc='upper right')


def plot_img_and_recon(output_dir, epoch_id, cmap='gray'):
    imgs_dir = os.path.join(output_dir, 'training', 'imgs')
    img_path = os.path.join(imgs_dir, 'epoch_%d_data.npy' % epoch_id)
    img = np.load(img_path)
    recon_path = os.path.join(imgs_dir, 'epoch_%d_recon.npy' % epoch_id)
    recon = np.load(recon_path)
    print('Shape of img data:')
    print(img.shape)
    print('Shape of recon:')
    print(recon.shape)

    fig, axes = plt.subplots(nrows=2, ncols=min(5, len(img)), figsize=(18, 4))
    i = 0
    for one_img, one_recon in zip(img, recon):
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
