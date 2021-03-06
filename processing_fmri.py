"""Data processing pipeline."""

import csv
from joblib import Parallel, delayed
import logging
import luigi
import numpy as np
import operator
import os
import re
import shutil
import skimage.transform

import imtk

import warnings
warnings.filterwarnings("ignore")


DEBUG = False

SESSIONS = [
    'ses001-022', 'ses023-045', 'ses046-068', 'ses069-091', 'ses092-107']
SELECTED_SESSIONS = 'ses069-091'

# Memory Bug
START_SES = 69
STOP_SES = 91

FETCH_FROM = 'open_neuro'

MY_CONNECTOME_ADDRESS = (
    's3://openneuro/ds000031/'
    'ds000031_R1.0.4/compressed/'
    'ds000031_R1.0.4_fmriprep_%s.zip' % SELECTED_SESSIONS)

IMG_WIDTH = 96
IMG_HEIGHT = 96
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT)
DEFAULT_FMRI_SHAPE = (128, 128, 52)
IMG_DIM = len(IMG_SHAPE)

FRAC_TEST = 0.1
FRAC_VAL = 0.2

NEURO_DIR = '/neuro'
BOLD_DIR = os.path.join(NEURO_DIR, 'boldscans')

CSV_COLS = ['path', 'ses', 'task', 'run', 'time']


class FetchMyConnectomeDataset(luigi.Task):
    """
    Download data from the My Connectome project.
    Data processed by fmriprep:
    https://legacy.openfmri.org/dataset/ds000031/
    """
    target_dir = os.path.join(BOLD_DIR, 'processed_4d')

    def requires(self):
        pass

    def fetch_from_open_fmri(self):
        logging.info(
            'Downloading %s sessions of My Connectome,'
            ' processed by fmriprep...' % SELECTED_SESSIONS)
        # target_path = os.path.join(
        #   BOLD_DIR, 'ds000031_R1.0.4_fmriprep_%s' % SELECTED_SESSIONS)
        os.system(
            "aws --no-sign-request s3 cp %s %s" %
            (MY_CONNECTOME_ADDRESS, BOLD_DIR))

        zip_path = os.path.join(
            BOLD_DIR, 'ds000031_R1.0.4_fmriprep_%s.zip' % SELECTED_SESSIONS)

        os.system("unzip %s -d tmp/" % zip_path)
        os.system(
            "mv tmp/ds000031_R1.0.4/derivatives/"
            "fmriprep_1.0.0/fmriprep/sub-01/* .")
        os.system("rm -r tmp/*")
        # TODO(nina): Put data in processed_4d format

    def fetch_from_open_neuro(self):
        if len(os.listdir(BOLD_DIR)) == 0:
            os.system(
                "aws s3 sync --no-sign-request "
                "s3://openneuro.org/ds000031 %s" % BOLD_DIR)

        sub_dir = os.path.join(BOLD_DIR, 'sub-01')
        modality = 'func'
        if not os.path.isdir(self.target_dir):
            logging.info('Make directory %s...' % self.target_dir)
            os.mkdir(self.target_dir)
            os.chmod(self.target_dir, 0o777)

        for ses_dir in os.listdir(sub_dir):
            ses_dir = os.path.join(sub_dir, ses_dir)
            if not os.path.isdir(ses_dir):
                continue
            for modality_dir in os.listdir(ses_dir):
                modality_dir = os.path.join(ses_dir, modality_dir)
                for filename in os.listdir(modality_dir):
                    path = os.path.join(modality_dir, filename)
                    if not re.search(modality, path):
                        continue
                    if not re.search('bold.nii.gz', path):
                        continue

                    target_path = os.path.join(self.target_dir, filename)
                    logging.info('Moving %s to %s...' % (path, target_path))
                    shutil.move(path, target_path)

    def run(self):
        if FETCH_FROM == 'open_fmri':
            self.fetch_from_open_fmri()
        elif FETCH_FROM == 'open_neuro':
            self.fetch_from_open_neuro()

    def output(self):
        return luigi.LocalTarget(self.target_dir)


class Process4Dto3D(luigi.Task):
    depth = int(DEFAULT_FMRI_SHAPE[2] * IMG_SHAPE[0] / 128)
    shape_str = '%dx%dx%d' % (IMG_SHAPE + (depth,))

    input_dir = os.path.join(BOLD_DIR, 'processed_4d')
    target_dir = os.path.join(BOLD_DIR, 'processed_%dd' % IMG_DIM)
    csv_path = os.path.join(target_dir, 'metadata.csv')

    def requires(self):
        return FetchMyConnectomeDataset()

    def process_file(self, path, output):
        if os.path.isdir(path):
            return

        basename = os.path.basename(path)
        _, ses, task, run, _ = basename.split('_')
        ses = int(ses.split('-')[1])
        task = task.split('-')[1]
        run = int(run.split('-')[1])

        if ses < START_SES or ses > STOP_SES:
            return

        logging.info('Processing 4D file %s' % path)

        prefix = basename.split('.')[0]

        array_4d = imtk.get_array_from_path(path)

        if array_4d.shape[:2] != IMG_SHAPE:
            array_4d = skimage.transform.resize(
                array_4d,
                (IMG_SHAPE[0], IMG_SHAPE[1],
                 self.depth, array_4d.shape[-1]))
        for i_img in range(array_4d.shape[-1]):
            img = array_4d[:, :, :, i_img]

            if IMG_DIM == 2:
                img = img[:, :, int(img.shape[2] / 2)]

            img_min = np.min(img)
            img_max = np.max(img)
            img = (img - img_min) / (img_max - img_min)

            img_basename = prefix + '_%d' % i_img + '.npy'
            img_path = os.path.join(self.target_dir, img_basename)
            logging.info('Saving %dD image %s' % (IMG_DIM, img_path))
            np.save(img_path, img)

            row = [img_path, ses, task, run, i_img]
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

        with open(self.csv_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_COLS)
            for row in output:
                writer.writerow(row)

    def output(self):
        return luigi.LocalTarget(self.csv_path)


class MakeDataset(luigi.Task):
    if IMG_DIM == 3:
        depth = int(DEFAULT_FMRI_SHAPE[2] * IMG_SHAPE[0] / 128)
        shape_str = '%dx%dx%d' % (IMG_SHAPE + (depth,))
    else:
        shape_str = '%dx%d' % IMG_SHAPE
    target_dir = os.path.join(NEURO_DIR, 'train_val_datasets')

    train_csv_path = os.path.join(
        target_dir, 'train_%s_%s_labels.csv' % ('fmri', shape_str))
    val_csv_path = os.path.join(
        target_dir, 'val_%s_%s_labels.csv' % ('fmri', shape_str))
    test_csv_path = os.path.join(
        target_dir, 'test_%s_%s_labels.csv' % ('fmri', shape_str))

    train_path = os.path.join(
            target_dir, 'train_%s_%s.npy' % ('fmri', shape_str))
    val_path = os.path.join(
            target_dir, 'val_%s_%s.npy' % ('fmri', shape_str))
    test_path = os.path.join(
            target_dir, 'test_%s_%s.npy' % ('fmri', shape_str))

    def requires(self):
        return Process4Dto3D()

    def write_first_row(self, csv_path):
        with open(csv_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_COLS)

    def run(self):
        csv_path = self.input().path

        with open(csv_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            sorted_rows = sorted(reader, key=operator.itemgetter(1))
            sess = []
            for row in sorted_rows:
                sess.append(row[1])
            n_ses = len(set(sess))

        logging.info('Creating train/val/test dataset of rfMRIs.')

        self.write_first_row(self.train_csv_path)
        self.write_first_row(self.val_csv_path)
        self.write_first_row(self.test_csv_path)

        i_ses = 0
        ses_previous = 'none'

        train_paths, val_paths, test_paths = [], [], []
        print(n_ses)
        for row in sorted_rows:
            path, ses, _, _, _ = row
            if path == 'path':
                continue

            if ses != ses_previous:
                i_ses += 1
                ses_previous = ses

            if i_ses < int((1 - FRAC_VAL - FRAC_TEST) * n_ses):
                with open(self.train_csv_path, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(row)
                train_paths.append(path)
            elif i_ses < int((1 - FRAC_TEST) * n_ses):
                with open(self.val_csv_path, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(row)
                val_paths.append(path)
            elif i_ses < n_ses:
                with open(self.test_csv_path, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(row)
                test_paths.append(path)
            else:
                break

        train_output = [np.load(one_path) for one_path in train_paths]
        val_output = [np.load(one_path) for one_path in val_paths]
        test_output = [np.load(one_path) for one_path in test_paths]

        train = np.asarray(train_output)
        val = np.asarray(val_output)
        test = np.asarray(test_output)

        train = np.expand_dims(train, axis=1)
        val = np.expand_dims(val, axis=1)
        test = np.expand_dims(test, axis=1)

        np.save(self.output()['train_fmri'].path, train)
        np.save(self.output()['val_fmri'].path, val)
        np.save(self.output()['test_fmri'].path, test)

    def output(self):
        return {'train_fmri':
                luigi.LocalTarget(self.train_path),
                'val_fmri':
                luigi.LocalTarget(self.val_path),
                'test_fmri':
                luigi.LocalTarget(self.test_path)
                }


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
