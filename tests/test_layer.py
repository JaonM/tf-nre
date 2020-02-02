import numpy as np
import tensorflow as tf

from tf_nre.layers import EncoderLayer, EntityAttentionLayer, CNNAttentionLayer


def test_encoder_layer():
    seq, e1_pos, e2_pos = tf.constant([[1, 2, 3, 4]]), tf.constant([[0, 1, 2, 3]]), tf.constant([[4, 5, 6, 7]])
    encoder = EncoderLayer(10, 3, 3, 4, 5)
    shape = encoder([seq, e1_pos, e2_pos]).shape
    assert shape == (1, 4, 48)


def test_entity_attention_layer():
    token_seq = tf.constant([[1, 2, 3, 4], [1, 2, 3, 4]])
    entity_seq = tf.ragged.constant([[2], [3, 4]])
    att = EntityAttentionLayer(3, 4, 5)
    weight = att([entity_seq, token_seq])  # tf.RaggedTensor
    print(weight.shape)
    print(weight)


def test_cnn_attention_layer():
    R = tf.constant(np.random.rand(3, 10, 48))
    label_emb = tf.constant(np.random.rand(19, 10))

    layer = CNNAttentionLayer(32, 3)
    out = layer([R, label_emb])
    assert out.shape == (3, 32)
