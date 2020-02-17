import numpy as np
import tensorflow as tf

from tf_nre.train import distance_fn


def test_distance_fn():
    predicted = tf.constant(np.random.rand(32, 128))
    label = tf.constant(np.random.rand(32, 128))
    d = distance_fn(predicted, label)
    assert d.shape == (32,)
