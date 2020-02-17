import tensorflow as tf

from tf_nre.model import MultiLevelAttCNN


def test_model():
    seq, e1_pos, e2_pos = tf.constant([[1, 2, 3, 4]]), tf.constant([[0, 1, 2, 3]]), tf.constant([[4, 5, 6, 7]])
    e1_seq, e2_seq = tf.constant([[1]]), tf.constant([[3, 4]])
    model = MultiLevelAttCNN(10, 3, 3, 5, 4, 8, 3, 19, 8)
    out = model([seq, e1_pos, e2_pos, e1_seq, e2_seq], training=True)
    assert out.shape == (1, 8)

    out = model([seq, e1_pos, e2_pos, e1_seq, e2_seq], training=False)
    assert out.shape == (1,)
