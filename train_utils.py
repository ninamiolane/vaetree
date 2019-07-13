"""Utils to factorize code for both pipelines."""

import glob
import logging
import os

import torch
import torch.nn as tnn

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


def init_training(train_dir, modules, optimizers):
    """Initialization: Load ckpts or xavier normal init."""
    start_epoch = 0
    train_losses_all_epochs = []
    val_losses_all_epochs = []

    path_base = os.path.join(
        train_dir, 'epoch_*_checkpoint.pth')
    ckpts = glob.glob(path_base)
    if len(ckpts) == 0:
        logging.info('No checkpoints found. Initializing with Xavier Normal.')
        for module in modules.values():
            module.apply(init_xavier_normal)
    else:
        ckpts_ids_and_paths = [
            (int(f.split('_')[3]), f) for f in ckpts]
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
