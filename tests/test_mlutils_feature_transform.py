import unittest
import pytest
import numpy as np
from mlutils.featuresengine import data_scaling, data_binning
from tests.config_tests import TEST_ROOT_DIR

@pytest.fixture()
def get_test_data():

    data = [float(i) for i in range(10)]
    data = np.array(data).reshape(-1, 1)
    return data


def test_invalid_scaling_strategy(get_test_data):
    with pytest.raises(ValueError):
        scaled_data = data_scaling(get_test_data, strategy='INVALID')

def test_min_max_scaling(get_test_data):
    scaled_data = data_scaling(get_test_data, strategy='min-max', feature_range=(0, 1))
    assert int(np.min(scaled_data)) == 0
    assert int(np.max(scaled_data)) == 1

def test_min_max_scaling_fail(get_test_data):

    with pytest.raises(ValueError):
        scaled_data = data_scaling(get_test_data, strategy='min-max')

def test_z_score_scaling(get_test_data):
    scaled_data = data_scaling(get_test_data, strategy='z-score')


def test_robust_scaling(get_test_data):
    scaled_data = data_scaling(get_test_data, strategy='robust')

def test_invalid_binning_strategy(get_test_data):
    with pytest.raises(ValueError):
        scaled_data = data_binning(get_test_data, n_bins=3, strategy='INVALID')

def test_uniform_data_binning(get_test_data):
    scaled_data = data_binning(get_test_data, n_bins=3, strategy='uniform', encode='ordinal')

def test_quantile_data_binning(get_test_data):
    scaled_data = data_binning(get_test_data, n_bins=3, strategy='quantile', encode='ordinal')

def test_kmeans_data_binning(get_test_data):
    scaled_data = data_binning(get_test_data, n_bins=3, strategy='kmeans', encode='ordinal')