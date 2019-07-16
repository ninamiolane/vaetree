"""Hyperparameter search."""

from _future_ import absolute_import
from _future_ import division
from _future_ import print_function

import config
import os
import sys
sys.path.append('/code/froglabs')  # NOQA
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from keras import backend as K
import click
import dataset
import keras
import logging
import numpy as np
import ray
import scipy.ndimage as ndimage
import training
import utils
import weather_api
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
logging.basicConfig(level=logging.INFO)


def limit_threads(num_threads):
    K.set_session(
        K.tf.Session(
            config=K.tf.ConfigProto(
                intra_op_parallelism_threads=num_threads,
                inter_op_parallelism_threads=num_threads)))


def get_best_trial(trial_list, metric):
    """Retrieve the best trial."""
    return max(trial_list, key=lambda trial: trial.last_result.get(metric, 0))


def get_sorted_trials(trial_list, metric):
    return sorted(
        trial_list,
        key=lambda trial: trial.last_result.get(metric, 0),
        reverse=True)


def get_best_result(trial_list, metric):
    """Retrieve the last result from the best trial."""
    return {metric: get_best_trial(trial_list, metric).last_result[metric]}


def get_best_model(model_creator, trial_list, metric):
    """Restore a model from the best trial."""
    sorted_trials = get_sorted_trials(trial_list, metric)
    for best_trial in sorted_trials:
        try:
            print('Creating model...')
            model = model_creator(**best_trial.config)
            weights = os.path.join(best_trial.logdir,
                                   best_trial.last_result['checkpoint'])
            print('Loading from', weights)
            model.load_weights(weights)
            break
        except Exception as e:
            print(e)
            print('Loading failed. Trying next model')
    return model


class TuneCallback(keras.callbacks.Callback):
    """Custom Callback for Tune."""

    def _init_(self, reporter):
        super(TuneCallback, self)._init_()
        self.reporter = reporter
        self.top_acc = -float('inf')
        self.last_10_results = []

    def on_epoch_end(self, batch, logs={}):
        """Reports the last result"""
        logging.info(logs)
        curr_acc = 1 - logs['val_loss']
        if curr_acc > self.top_acc:
            self.model.save_weights('weights_tune_tmp.h5')
            os.rename('weights_tune_tmp.h5', 'weights_tune.h5')
            self.top_acc = curr_acc

        self.reporter(
            mean_accuracy=curr_acc,
            checkpoint='weights_tune.h5')


class GoodError(Exception):
    pass


def test_reporter(train_mnist_tune):
    def mock_reporter(**kwargs):
        assert 'mean_accuracy' in kwargs, 'Did not report proper metric'
        assert 'checkpoint' in kwargs, 'Accidentally removed `checkpoint`?'
        raise GoodError('This works.')

    try:
        train_mnist_tune({}, mock_reporter)
    except TypeError as e:
        print('Forgot to modify function signature?')
        raise e
    except GoodError:
        print('Works!')
        return 1
    raise Exception('Didn\'t call reporter...')


def prepare_data(data):
    try:
        new_data = np.array(data).reshape((1, 28, 28, 1)).astype(np.float32)
    except ValueError as e:
        print('Try running this notebook in `jupyter notebook`.')
        raise e
    return ndimage.gaussian_filter(new_data, sigma=(0.5))


def train(conf, reporter):
    y_file = '%s/germany_power_MW_from_2015-01-01_to_2018-11-14.csv' % config.HISTORICAL_DIR  # noqa
    y_ts = dataset.load_csv(y_file, 'ts', 'power')
    variables = [[weather_api.Variable('GHI', 2),
                  weather_api.Variable('temperature', 3),
                  weather_api.Variable('TCWV', 3),
                  weather_api.Variable('SNOD', 3),
                  weather_api.Variable('clearsky', 0)]]
    ds = dataset.Dataset(
        y_ts, variables[0], valid_time_hours=[3, 20], save=True)

    params = {'lr': conf['lr'],
              'decay': conf['decay'],
              'seed': 100,
              'batch_size': conf['batch_size'],
              'window_size': 10,
              'valid_time_hours': (3, 20),
              'variables': variables[0],
              'epochs': 50,
              'norm_method': 'scaler',
              'loss': 'mean_squared_error',
              'model_type': 'pvnet',
              'tag': '2017 test',
              'image_shape': (32, 32),
              }
    logging.info(params)
    training_id = utils.get_training_id(params)
    callbacks = [TuneCallback(reporter)]
    params = training.train(
        ds, params, training_id=training_id, callbacks=callbacks)
    logging.info('RMSE: %.02f  nRMSE:%.02f%%' % (params['rmse'],
                                                 params['nrmse']))
    logging.info('Training done. %s' % training_id)
    return {"mean_accuracy": 1 - params['nrmse'] / 100.0}


@click.command()
@click.option('--smoke_test', default=False, help='smoke test')
def search(smoke_test):
    space = {
        'lr': hp.loguniform("lr", np.log(10)-6, np.log(10)-3),
        'decay': hp.loguniform("decay", np.log(10)-6, np.log(10)-3),
        'batch_size': 24,
    }
    training_spec = {
        "num_samples": 4,
        'stop': {
            "mean_accuracy": 4,
            "training_iteration": 40
        }
    }

    ray.init(redis_address="95.216.70.217:6379", redis_password="zohl9Phi")
    hyperband = AsyncHyperBandScheduler(
        time_attr='training_iteration', reward_attr='mean_accuracy')
    hyperopt_search = HyperOptSearch(space, reward_attr="mean_accuracy")
    results = tune.run(
        train,
        name='pv',
        search_alg=hyperopt_search,
        scheduler=hyperband,
        resources_per_trial={
            'cpu': 8,
            'gpu': 1,
        },
        **training_spec)
    logging.info("The best result is", get_best_result(
        results, metric="mean_accuracy"))


if _name_ == '_main_':
    search()
