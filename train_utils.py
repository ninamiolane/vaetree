"""Utils to factorize code for both pipelines."""

import glob
import logging
import os

import torch
import torch.nn as tnn

import nn

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

CKPT_PERIOD = 1


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


def init_custom(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_function(weights_init='xavier'):
    if weights_init == 'xavier':
        return init_xavier_normal
    elif weights_init == 'kaiming':
        return init_kaiming_normal
    elif weights_init == 'custom':
        return init_custom
    else:
        raise NotImplementedError(
            'This weight initialization is not implemented.')


def init_modules_and_optimizers(nn_architecture, train_params):
    modules = {}
    optimizers = {}

    nn_type = nn_architecture['nn_type']
    img_shape = nn_architecture['img_shape']
    latent_dim = nn_architecture['latent_dim']
    data_dim = nn_architecture['data_dim']

    lr = train_params['lr']
    beta1 = train_params['beta1']
    beta2 = train_params['beta2']

    assert nn_type in ['linear', 'conv', 'gan']

    # Modules
    if nn_type == 'linear':
        vae = nn.Vae(
            latent_dim=latent_dim,
            data_dim=data_dim).to(DEVICE)
    elif nn_type == 'conv':
        vae = nn.VaeConv(
                latent_dim=latent_dim,
                img_shape=img_shape,
                spd=nn_architecture['spd']).to(DEVICE)
    else:
        vae = nn.VaeGan(
            latent_dim=latent_dim,
            img_shape=img_shape).to(DEVICE)
        discriminator = nn.DiscriminatorGan(
            latent_dim=latent_dim,
            img_shape=img_shape).to(DEVICE)
        modules['discriminator_reconstruction'] = discriminator

    modules['encoder'] = vae.encoder
    modules['decoder'] = vae.decoder

    # Optimizers
    optimizers['encoder'] = torch.optim.Adam(
        modules['encoder'].parameters(), lr=lr)
    optimizers['decoder'] = torch.optim.Adam(
        modules['decoder'].parameters(), lr=lr, betas=(beta1, beta2))

    if nn_type == 'gan':
        optimizers['discriminator_reconstruction'] = torch.optim.Adam(
            modules['discriminator_reconstruction'].parameters(),
            lr=lr,
            betas=(beta1, beta2))

    return modules, optimizers


def init_training(train_dir, nn_architecture, train_params):
    """Initialization: Load ckpts or init."""
    start_epoch = 0
    train_losses_all_epochs = []
    val_losses_all_epochs = []

    modules, optimizers = init_modules_and_optimizers(
        nn_architecture, train_params)

    path_base = os.path.join(train_dir, 'epoch_*_checkpoint.pth')
    ckpts = glob.glob(path_base)
    if len(ckpts) == 0:
        weights_init = train_params['weights_init']
        logging.info(
            'No checkpoints found. Initializing with %s.' % weights_init)
        for module in modules.values():
            module.apply(init_function(weights_init))

    else:
        ckpts_ids_and_paths = [
            (int(f.split('_')[-2]), f) for f in ckpts]
        ckpt_id, ckpt_path = max(
            ckpts_ids_and_paths, key=lambda item: item[0])
        logging.info('Found checkpoints. Initializing with %s.' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        for module_name in modules.keys():
            module = modules[module_name]
            optimizer = optimizers[module_name]
            module_ckpt = ckpt[module_name]
            module.load_state_dict(module_ckpt['module_state_dict'])
            optimizer.load_state_dict(
                module_ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            train_losses_all_epochs = ckpt['train_losses']
            val_losses_all_epochs = ckpt['val_losses']

    return (modules, optimizers, start_epoch,
            train_losses_all_epochs, val_losses_all_epochs)


def save_checkpoint(epoch, modules, optimizers, dir_path,
                    train_losses_all_epochs, val_losses_all_epochs,
                    nn_architecture):
    checkpoint = {}
    for module_name in modules.keys():
        module = modules[module_name]
        optimizer = optimizers[module_name]
        checkpoint[module_name] = {
            'module_state_dict': module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
        checkpoint['epoch'] = epoch
        checkpoint['train_losses'] = train_losses_all_epochs
        checkpoint['val_losses'] = val_losses_all_epochs
        checkpoint['nn_architecture'] = nn_architecture

    checkpoint_path = os.path.join(
        dir_path, 'epoch_%d_checkpoint.pth' % epoch)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(output, algo_name='vae', epoch_id=None):
    if epoch_id is None:
        ckpts = glob.glob(
            '%s/train_%s/epoch_*_checkpoint.pth' % (
                output, algo_name))
        if len(ckpts) == 0:
            raise ValueError('No checkpoints found.')
        else:
            ckpts_ids_and_paths = [(int(f.split('_')[-2]), f) for f in ckpts]
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
    ckpt = load_checkpoint(
        output=output, algo_name=algo_name, epoch_id=epoch_id)
    nn_architecture = ckpt['nn_architecture']

    nn_type = nn_architecture['nn_type']
    print('Loading %s from network of architecture: %s...' % (
        module_name, nn_type))
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
