import tensorflow as tf

from tf_nre.model import MultiLevelAttCNN
from tf_nre.main import compute_label_emb_size, L2_PARAM


def test_model():
    seq, e1_pos, e2_pos = tf.constant([[1, 2, 3, 4]]), tf.constant([[0, 1, 2, 3]]), tf.constant([[4, 5, 6, 7]])
    e1_seq, e2_seq = tf.constant([[1]]), tf.constant([[3, 4]])
    label_emb_size = compute_label_emb_size(4, 3)
    model = MultiLevelAttCNN(10, 3, 3, 5, 4, 8, 3, 19, label_emb_size, L2_PARAM)
    out = model([seq, e1_pos, e2_pos, e1_seq, e2_seq], training=True)
    assert out.shape == (1, label_emb_size)

    out = model([seq, e1_pos, e2_pos, e1_seq, e2_seq], training=False)
    assert out.shape == (1,)
