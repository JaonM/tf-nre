# training process
# TODO loss function
import tensorflow as tf


def loss_fn(predicted, label, label_emb):
    """
    loss function
    :param predicted: tensor (batch_size,num_filter)
    :param label: tensor (batch_size,)
    :param label_emb: tensor (label_size,label_dim)
    :return:
    """
    pass


def distance_fn(predicted, label_tensor):
    """
    distance function
    :param predicted: (batch_size,num_filter)
    :param label_tensor: (batch_size,label_dim)
    label_dim == num_filter
    :return: (batch_size,)
    """
    predicted = predicted / tf.norm(predicted, axis=1, keepdims=True)
    return tf.norm(predicted - label_tensor, axis=1)
