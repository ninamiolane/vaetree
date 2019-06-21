""" Process T1 mri."""

from joblib import Parallel, delayed
import glob
import logging
import luigi
import os

import sklearn.model_selection

import numpy as np

import imtk

DEBUG = False

NEURO_DIR = '/neuro'
T1SCANS_DIR = os.path.join(NEURO_DIR, 't1scans')
SAVED_DIR = '/neuro/preprocessed_saved'
NII_PATHS = './datasets/openneuro_files.txt'

# TODO(nina): Put this in Singularity receipe
os.environ['ANTSPATH'] = '/usr/lib/ants/'

DATASET_NAME = 'all'

IMG_SHAPE = (128, 128)
IMG_3D_SHAPE = (64, 128, 128)
IMG_DIM = len(IMG_SHAPE)

DATA_TYPE = 'mri'
FRAC_VAL = 0.2

ANTS_PROCESSING = False

if DEBUG:
    N_FILEPATHS = 10


class FetchOpenNeuroDataset(luigi.Task):
    """
    Download all t1 weighted .nii.gz from OpenNeuro.
    Save them on gne.
    """

    file_list_path = NII_PATHS
    target_dir = T1SCANS_DIR

    def dl_file(self, path):
        path = path.strip()
        target_path = self.target_dir + os.path.dirname(path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # TODO(nina): Replace with subprocess
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


class SelectGood3D(luigi.Task):
    """
    Deletes images if:
    - bad affine orientation
    - if the background is a white noise
    """

    target_dir = os.path.join(NEURO_DIR, 'selected')

    def requires(self):
        return {'dataset': FetchOpenNeuroDataset()}

    def process_file(self, path, processed_files):
        logging.info('Loading image %s...', path)

        if imtk.has_bad_affine_orientation(path):
            logging.info(
                'Skip image %s - bad affine orientation' % path)
            return
        if imtk.has_white_noise_background(path):
            logging.info(
                'Skip image %s - white noise background' % path)
            return

        path_split = path.split('/')
        target_name = path_split[3] + '_' + path_split[-1]
        target_path = os.path.join(self.target_dir, target_name)
        os.system('cp %s %s' % (path, target_path))

        processed_files.append(path)

    def run(self):
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        directory = self.input()['dataset'].path
        filepaths = glob.glob(directory + '/**/*.nii.gz', recursive=True)

        if DEBUG:
            filepaths = filepaths[:N_FILEPATHS]

        logging.info(
            'Selecting from %d 3D raw nii filepaths...' % len(filepaths))

        processed_files = []
        Parallel(backend="threading", n_jobs=4)(
            delayed(self.process_file)(f, processed_files)
            for f in filepaths)

        logging.info(
            'Kept %d/%d 3D raw nii filepaths.' % (
                len(processed_files), len(filepaths)))

    def output(self):
        return luigi.LocalTarget(self.target_dir)


class Process3D(luigi.Task):
    """
    if ANTS_PROCESSING:

    Performs the following:
    - N4BiasFieldCorrection
    - Atropos for segmentation and brain extraction, which uses:
    --- antsRegistration
    --- antsApplyTransforms
    # -k parameter to keep the temporary files, i.e. the segmentation
    # -z parameter > 0 runs a debug version, 10min/nii instead of 20min/nii
    From:
    https://github.com/ANTsX/ANTs/blob/master/Scripts/antsBrainExtraction.sh

    Labels in array after segmentation:
    - Background=0., CSF=1., GM=2., WM=3.
    """
    target_dir = os.path.join(NEURO_DIR, 'preprocessed')
    brain_template_with_skull = os.path.join(
        NEURO_DIR, 'T_template0.nii.gz')
    brain_prior = os.path.join(
        NEURO_DIR, 'T_template0_BrainCerebellumProbabilityMask.nii.gz')
    registration_mask = os.path.join(
        NEURO_DIR, 'T_template0_BrainCerebellumRegistrationMask.nii.gz')

    def requires(self):
        return {'dataset': SelectGood3D()}

    def process_file(self, path, output):
        # TODO(nina): Replace os.system by subprocess and control ANTs verbose
        # TODO(nina): Put a progress bar?
        logging.info('Loading image %s...', path)

        basename = os.path.basename(path)
        saved_path = os.path.join(SAVED_DIR, 'mri_' + basename)
        target_path = os.path.join(self.target_dir, 'mri_' + basename)
        seg_saved_path = os.path.join(SAVED_DIR, 'seg_' + basename)
        seg_target_path = os.path.join(self.target_dir, 'seg_' + basename)
        delete_tmp = False
        if not os.path.exists(saved_path):
            logging.info('No saved processed nii found. Processing...')
            tmp_prefix = imtk.get_tmpfile_prefix()

            os.system(
                '/usr/lib/ants/antsBrainExtraction.sh'
                ' -d {} -a {} -e {} -m {} -f {} -o {} -k 1 -z {}'.format(
                    3,
                    path,
                    self.brain_template_with_skull,
                    self.brain_prior,
                    self.registration_mask,
                    tmp_prefix,
                    int(DEBUG)))

            img_tmp_path = tmp_prefix + 'BrainExtractionBrain.nii.gz'
            os.system('mv %s %s' % (img_tmp_path, saved_path))

            seg_tmp_path = tmp_prefix + 'BrainExtractionSegmentation.nii.gz'
            os.system('mv %s %s' % (seg_tmp_path, seg_saved_path))
            delete_tmp = True

        os.system('cp %s %s' % (saved_path, target_path))
        os.system('cp %s %s' % (seg_saved_path, seg_target_path))

        output.append(target_path)

        if delete_tmp:
            tmp_paths = glob.glob(tmp_prefix + '*.nii.gz')
            for path in tmp_paths:
                print('Removing %s...' % path)
                os.remove(path)

    def run(self):
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        directory = self.input()['dataset'].path
        input_paths = glob.glob(directory + '/*.nii.gz', recursive=True)

        if DEBUG:
            input_paths = input_paths[:min(len(input_paths), N_FILEPATHS)]

        n_input_paths = len(input_paths)
        logging.info(
            'Processing %d 3D raw nii input_paths' % n_input_paths)

        processed_input_paths = []
        Parallel(backend="threading", n_jobs=4)(
            delayed(self.process_file)(f, processed_input_paths)
            for f in input_paths)

        logging.info('Processed %d/%d nii input_paths.' % (
            len(processed_input_paths), n_input_paths))

    def output(self):
        return luigi.LocalTarget(self.target_dir)


class MakeDataSet(luigi.Task):
    """
    Resize images / segmentations.
    Extract slices if IMG_DIM is set to 2D, taking care to separate patients.
    """
    shape_str = '%dx%dx%d' % IMG_SHAPE if IMG_DIM == 3 else '%dx%d' % IMG_SHAPE
    target_dir = os.path.join(NEURO_DIR, 'train_val_datasets')

    train_path = os.path.join(
            target_dir, 'train_%s_%s.npy' % (DATA_TYPE, shape_str))
    val_path = os.path.join(
            target_dir, 'val_%s_%s.npy' % (DATA_TYPE, shape_str))

    random_state = 13

    def requires(self):
        return {'dataset': Process3D()}

    def run(self):
        if not os.path.isdir(self.target_dir):
            os.mkdir(self.target_dir)
            os.chmod(self.target_dir, 0o777)

        directory = self.input()['dataset'].path
        input_paths = glob.glob(directory + '/%s_*.nii.gz' % DATA_TYPE)
        if DEBUG:
            input_paths = input_paths[:min(len(input_paths), N_FILEPATHS)]

        logging.info(
            'Creating 3D dataset from %d nii paths.' % len(input_paths))
        output = []
        Parallel(backend="threading", n_jobs=4)(
            delayed(imtk.extract_resize_3d)(f, output, IMG_3D_SHAPE)
            for f in input_paths)
        array = np.asarray(output)
        shape_with_channels = (array.shape[0],) + (1,) + array.shape[1:]
        array = array.reshape(shape_with_channels)

        if IMG_DIM == 2:
            logging.info('Slicing to 2D dataset.')

            output = []
            Parallel(backend="threading", n_jobs=4)(
                delayed(imtk.slice_to_2d)(one_array, output)
                for one_array in array)
            array = np.asarray(output)

        logging.info('Normalizing intensities to [0, 1].')
        output = []
        Parallel(backend="threading", n_jobs=4)(
            delayed(imtk.normalize_intensities)(one_array, output)
            for one_array in array)
        array = np.asarray(output)

        logging.info('Splitting into train/val.')

        split = sklearn.model_selection.train_test_split(
            array,
            test_size=FRAC_VAL,
            random_state=self.random_state)
        train, val = split

        np.save(self.output()['train_' + DATA_TYPE].path, train)
        np.save(self.output()['val_' + DATA_TYPE].path, val)

    def output(self):
        return {'train_' + DATA_TYPE:
                luigi.LocalTarget(self.train_path),
                'val_' + DATA_TYPE:
                luigi.LocalTarget(self.val_path)}


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
