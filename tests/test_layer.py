import tensorflow as tf

from tf_nre.layers import EncoderLayer, EntityAttentionLayer


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
