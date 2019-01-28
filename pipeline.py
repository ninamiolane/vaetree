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
import sklearn.model_selection


HOME_DIR = '/scratch/users/nmiolane'


class FetchDataSet(luigi.Task):
    path = os.path.join(HOME_DIR, 'dataset')

    def requires(self):
        pass

    def run(self):
        pass

    def output(self):
        return luigi.LocalTarget(self.path)


class MakeDataSet(luigi.Task):
    path = os.path.join(HOME_DIR, 'output')
    first_slice = 42
    last_slice = 162
    test_fraction = 0.2

    def requires(self):
        return {'dataset': FetchDataSet()}

    def run(self):
        path = self.input()['dataset'].path
        filepaths = glob.glob(path + '/sub*T1w.nii.gz')
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
        imgs = imgs.reshape(imgs.shape + (1,))

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
            imgs, imgs, test_size=self.test_fraction, random_state=13)
        train_input, test_input, train_gt, test_gt = split

    def output(self):
        return luigi.LocalTarget(self.path)


class RunAll(luigi.Task):
    def requires(self):
        return MakeDataSet()

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
