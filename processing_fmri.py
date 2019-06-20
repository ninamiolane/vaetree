"""Data processing pipeline."""

import jinja2
from joblib import Parallel, delayed
import logging
import luigi
import numpy as np
import os
import skimage.transform

import imtk

import warnings
warnings.filterwarnings("ignore")


DEBUG = False

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT)
DEFAULT_FMRI_SHAPE = (128, 128, 52)
IMG_DIM = len(IMG_SHAPE)
N_SES_DEBUG = 3

FRAC_TEST = 0.1
FRAC_VAL = 0.2
NEURO_DIR = '/neuro'

LOADER = jinja2.FileSystemLoader('./templates/')
TEMPLATE_ENVIRONMENT = jinja2.Environment(
    autoescape=False,
    loader=LOADER)
TEMPLATE_NAME = 'report.jinja2'


class Process4Dto3D(luigi.Task):
    depth = int(DEFAULT_FMRI_SHAPE[2] * IMG_SHAPE[0] / 128)
    shape_str = '%dx%dx%d' % (IMG_SHAPE + (depth,))
    # Only rfMRI here, other nii have been removed:
    # (Get them back from open neuro)
    input_dir = '/neuro/boldscans/processed_4d/'
    target_dir = '/neuro/boldscans/processed_3d/'
    csv_path = os.path.join(target_dir, 'metadata.csv')

    def requires(self):
        pass

    def process_file(self, path, output):
        if os.path.isdir(path):
            return

        logging.info('Processing 4D file %s' % path)
        array_4d = imtk.get_array_from_path(path)

        if array_4d.shape[:2] != IMG_SHAPE:
            array_4d = skimage.transform.resize(
                array_4d,
                (IMG_SHAPE[0], IMG_SHAPE[1],
                 self.depth, array_4d.shape[-1]))

        array_4d_min = np.min(array_4d)
        array_4d_max = np.max(array_4d)
        array_4d = (array_4d - array_4d_min) / (
            array_4d_max - array_4d_min)

        basename = os.path.basename(path)
        prefix = basename.split('.')[0]

        for i_img_3d in range(array_4d.shape[-1]):
            img_3d = array_4d[:, :, :, i_img_3d]

            img_3d_basename = prefix + '_%d' % i_img_3d + '.npy'
            img_3d_path = os.path.join(self.target_dir, img_3d_basename)
            np.save(img_3d_path, img_3d)

            row = [img_3d_path, i_img_3d]
            output.append(row)

    def run(self):
        if not os.path.isdir(self.target_dir):
            os.mkdir(self.target_dir)
            os.chmod(self.target_dir, 0o777)

        # TODO(nina): Solve MemoryBug here
        logging.info('Extracting 3D images from rfMRI time-series.')

        output = []
        Parallel(backend="threading", n_jobs=4)(
            delayed(self.process_file)(
                os.path.join(self.input_dir, relpath),
                output)
            for relpath in os.listdir(self.input_dir))

        with open(self.csv_path, 'w') as csv:
            csv.write(self.labels)
            for row in output:
                csv.write(row)

    def output(self):
        return {'metadata':
                luigi.LocalTarget(self.csv_path)}


class MakeDataset(luigi.Task):
    if IMG_DIM == 3:
        depth = int(DEFAULT_FMRI_SHAPE[2] * IMG_SHAPE[0] / 128)
        shape_str = '%dx%dx%d' % (IMG_SHAPE + (depth,))
    else:
        shape_str = '%dx%d' % IMG_SHAPE
    target_dir = os.path.join(NEURO_DIR, 'train_val_datasets')

    train_path = os.path.join(
            target_dir, 'train_%s_%s.npy' % ('fmri', shape_str))
    val_path = os.path.join(
            target_dir, 'val_%s_%s.npy' % ('fmri', shape_str))

    def requires(self):
        return Process4Dto3D()

    def run(self):
        # If IMG_DIM == 3, then Process4Dto3D is enough
        if IMG_DIM == 2:
            train_3d_path = self.input()['train_%s' % DATA_TYPE].path
            val_3d_path = self.input()['val_%s' % DATA_TYPE].path
            train_3d = np.load(train_3d_path)
            val_3d = np.load(val_3d_path)

            logging.info('Creating 2D dataset of rfMRIs.')

            train_output = []
            Parallel(backend="threading", n_jobs=4)(
                delayed(imtk.slice_to_2d)(
                    one_train, train_output, AXIS[DATA_TYPE])
                for one_train in train_3d)
            val_output = []
            Parallel(backend="threading", n_jobs=4)(
                delayed(imtk.slice_to_2d)(
                    one_val, val_output, AXIS[DATA_TYPE])
                for one_val in val_3d)

            train_2d = np.asarray(train_output)
            val_2d = np.asarray(val_output)
            train = train_2d
            val = val_2d
            np.save(self.output()['train_fmri'].path, train)
            np.save(self.output()['val_fmri'].path, val)

    def output(self):
        return {'train_fmri':
                luigi.LocalTarget(self.train_path),
                'val_fmri':
                luigi.LocalTarget(self.val_path)}


class RunAll(luigi.Task):
    def requires(self):
        return MakeDataset()

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
