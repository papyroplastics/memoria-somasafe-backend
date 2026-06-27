import random
import numpy as np
import tensorflow as tf

from common.config import SEED

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
