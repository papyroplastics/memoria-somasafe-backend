import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def seeded():
    tf.random.set_seed(0)
