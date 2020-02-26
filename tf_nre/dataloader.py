# Parse serialized data from tf-record file
import tensorflow as tf

from tf_nre.train import MAX_LEN


def parse_single_train(example):
    features = {
        'text_seq': tf.io.FixedLenFeature((MAX_LEN,), tf.int64),
        'e1_seq': tf.io.VarLenFeature(tf.int64),
        'e2_seq': tf.io.VarLenFeature(tf.int64),
        'rel_e1_pos': tf.io.FixedLenFeature((MAX_LEN,), tf.int64),  # Need be changed to RaggedTensor after batching
        'rel_e2_pos': tf.io.FixedLenFeature((MAX_LEN,), tf.int64),
        'label': tf.io.FixedLenFeature((1,), tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example, features)
    return parsed_example


def parse_single_test(example):
    features = {
        'text_seq': tf.io.FixedLenFeature((MAX_LEN,), tf.int64),
        'e1_seq': tf.io.VarLenFeature(tf.int64),
        'e2_seq': tf.io.VarLenFeature(tf.int32),
        'rel_e1_pos': tf.io.FixedLenFeature((MAX_LEN,), tf.int64),
        'rel_e2_pos': tf.io.FixedLenFeature((MAX_LEN,), tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, features)
    return parsed_example


def get_dataset(filename, train=True):
    dataset = tf.data.TFRecordDataset(filename)
    if train:
        dataset = dataset.map(parse_single_train)
    elif not train:
        dataset = dataset.map(parse_single_test)
    return dataset


if __name__ == '__main__':
    dataset = get_dataset('data/input/train.tfrecord')
    print(dataset)
