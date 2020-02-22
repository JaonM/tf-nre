import numpy as np
import tensorflow as tf

from tf_nre.train import distance_fn, locate_farthest_label


def test_distance_fn():
    predicted = tf.constant(np.random.rand(32, 128))
    label = tf.constant(np.random.rand(32, 128))
    d = distance_fn(predicted, label)
    assert d.shape == (32,)


def test_locate_farthest_label():
    predicted = tf.constant(np.random.rand(32, 128))
    label_emb = tf.constant(np.random.rand(19, 128))
    predicted = locate_farthest_label(predicted, label_emb)
    print(predicted.shape)
    assert predicted.shape == (32, 128)
