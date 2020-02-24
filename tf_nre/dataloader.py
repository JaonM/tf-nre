# Parse serialized data from tf-record file
import tensorflow as tf
from tensorflow.data import TFRecordDataset

from tf_nre.train import MAX_LEN


def parse_single_train(example):
    features = {
        'text_seq': tf.io.FixedLenFeature((MAX_LEN,), tf.int32),
        'e1_seq': tf.io.VarLenFeature(tf.int32),
        'e2_seq': tf.io.VarLenFeature(tf.int32),
        'rel_e1_pos': tf.io.FixedLenFeature((MAX_LEN,), tf.int32),
        'rel_e2_pos': tf.io.FixedLenFeature((MAX_LEN,), tf.int32),
        'label': tf.io.FixedLenFeature((1,), tf.int32)
    }
    parsed_example = tf.io.parse_single_example(example, features)
    parsed_example['e1_seq'] = tf.RaggedTensor.from_sparse(parsed_example['e1_seq'])
    parsed_example['e2_seq'] = tf.RaggedTensor.from_sparse(parsed_example['e2_seq'])
    return parsed_example


def parse_single_test(example):
    features = {
        'text_seq': tf.io.FixedLenFeature((MAX_LEN,), tf.int32),
        'e1_seq': tf.io.VarLenFeature(tf.int32),
        'e2_seq': tf.io.VarLenFeature(tf.int32),
        'rel_e1_pos': tf.io.FixedLenFeature((MAX_LEN,), tf.int32),
        'rel_e2_pos': tf.io.FixedLenFeature((MAX_LEN,), tf.int32),
    }
    parsed_example = tf.io.parse_single_example(example, features)
    parsed_example['e1_seq'] = tf.RaggedTensor.from_sparse(parsed_example['e1_seq'])
    parsed_example['e2_seq'] = tf.RaggedTensor.from_sparse(parsed_example['e2_seq'])
    return parsed_example


def get_dataset(filename, train=True):
    dataset = TFRecordDataset(filename)
    if train:
        dataset = dataset.map(parse_single_train)
    elif not train:
        dataset = dataset.map(parse_single_test)
    return dataset
