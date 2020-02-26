# training process
import tensorflow as tf

# Hyper Parameters
MAX_LEN = 100
L2_PARAM = 0.001
LEARNING_RATE = 0.03
CONV_SIZE = 1000
KERNEL_SIZE = 4
POS_EMB_SIZE = 25


def loss_fn(predicted, label, label_emb):
    """
    loss function
    :param predicted: tensor (batch_size,num_filter)
    :param label: tensor (batch_size,)
    :param label_emb: tensor (label_size,label_dim)
    :return: (batch_size)
    """
    y = locate_label_dim(label, label_emb)
    y_f = locate_farthest_label(predicted, label_emb)
    l = 1 + distance_fn(predicted, y) - distance_fn(predicted, y_f)
    return l


def locate_label_dim(label, label_emb):
    """

    :param label: (batch_size,)
    :param label_emb: (label_size,label_dim)
    :return: (batch_size,label_dim)
    """
    label = tf.concat(list(map(lambda x: tf.expand_dims(label_emb[x], axis=0), label.numpy())), axis=0)
    return label


def locate_farthest_label(predicted, label_emb):
    """
    Find the farthest label according to distance function
    :param predicted: (batch_size,num_filter)
    :param label: (batch_size,)
    :param label_emb: (label_size,label_dim)
    :return:
    """

    def _inner_fn(_predicted, _label_emb):
        _predicted = tf.expand_dims(_predicted, axis=0)
        exp_label_emb = tf.expand_dims(_label_emb, axis=1)
        dist = tf.map_fn(lambda x: distance_fn(_predicted, x), exp_label_emb)  # (label_size,1)
        dist = tf.squeeze(dist)
        far_label = tf.argmax(dist)
        return _label_emb[far_label]  # (label_dim,)

    output = tf.map_fn(lambda x: _inner_fn(x, label_emb), predicted)
    return output  # (batch_size, label_dim)


def distance_fn(predicted, label_tensor):
    """
    distance function
    :param predicted: (batch_size,num_filter)
    :param label_tensor: (batch_size,label_dim)
    label_dim == num_filter
    :return: (batch_size,)
    """
    label_tensor = label_tensor / tf.norm(label_tensor, axis=1, keepdims=True)
    predicted = predicted / tf.norm(predicted, axis=1, keepdims=True)
    return tf.norm(predicted - label_tensor, axis=1)
