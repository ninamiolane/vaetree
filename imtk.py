"""Image TK."""

import os
import tempfile

import nibabel
import numpy as np
import skimage


def get_array_from_path(path):
    img = nibabel.load(path)
    array = img.get_fdata()
    array = np.nan_to_num(array)
    return array


def get_tmpfile_prefix(some_id='gne007_'):
    return os.path.join(
        tempfile.gettempdir(),
        next(tempfile._get_candidate_names()) + "_" + some_id)


def has_white_noise_background(path):
    array = get_array_from_path(path)
    std = np.std(array.reshape(-1))
    array = array / std
    mean = np.mean(array.reshape(-1))
    # HACK Alert
    # This is a way to check if the backgound is a white noise.
    if mean > 1.0:
        return True
    return False


def is_diag(mat):
    return np.all(mat == np.diag(np.diagonal(mat)))


def affine_matrix_permutes_axes(affine_matrix):
    mat = affine_matrix[:3, :3]
    if not is_diag(mat):
        return True
    if np.any(mat < 0):
        return True
    return False


def has_bad_affine_orientation(path):
    img = nibabel.load(path)

    if affine_matrix_permutes_axes(img.affine):
        return True
    return False


def extract_resize_3d(path, output, img_3d_shape):
    # TODO(nina): investigate distribution of sizes in datasets
    # TODO(nina): add DatasetReport Task
    array = get_array_from_path(path)
    # TODO(nina): Need to normalize/resample intensity histograms?
    if array.shape[1] != array.shape[2]:
        # This assumes that the shape is of the form (128, 256, 256)
        # square at the end
        print('Skip: non square shape of dim 1 and 2.')
        return
    array = skimage.transform.resize(array, img_3d_shape)
    output.append(array)


def normalize_intensities(array, output):
    min_array = np.min(array)
    max_array = np.max(array)
    array = (array - min_array) / (max_array - min_array)
    output.append(array)


def slice_to_2d(array, output, axis=3):
    if len(array.shape) != 4:
        # Adding channels
        array = np.expand_dims(array, axis=0)
    size = array.shape[axis]
    start = int(0.45 * size)
    end = int(0.55 * size)

    for k in range(start, end):
        img = array.take(indices=k, axis=axis)
        output.append(img)
