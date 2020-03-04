# Parse serialized data from tf-record file
import tensorflow as tf


class DataLoader(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def parse_single_train(self, example):
        features = {
            'text_seq': tf.io.FixedLenFeature((self.max_len,), tf.int64),
            # Need be changed to RaggedTensor after batching
            'e1_seq': tf.io.VarLenFeature(tf.int64),
            'e2_seq': tf.io.VarLenFeature(tf.int64),
            'rel_e1_pos': tf.io.FixedLenFeature((self.max_len,), tf.int64),
            'rel_e2_pos': tf.io.FixedLenFeature((self.max_len,), tf.int64),
            'label': tf.io.FixedLenFeature((1,), tf.int64)
        }

        parsed_example = tf.io.parse_single_example(example, features)
        return parsed_example

    def parse_single_test(self, example):

        features = {
            'text_seq': tf.io.FixedLenFeature((self.max_len,), tf.int64),
            'e1_seq': tf.io.VarLenFeature(tf.int64),
            'e2_seq': tf.io.VarLenFeature(tf.int32),
            'rel_e1_pos': tf.io.FixedLenFeature((self.max_len,), tf.int64),
            'rel_e2_pos': tf.io.FixedLenFeature((self.max_len,), tf.int64),
        }
        parsed_example = tf.io.parse_single_example(example, features)
        return parsed_example

    def get_dataset(self, filename, train=True):
        dataset = tf.data.TFRecordDataset(filename)
        if train:
            dataset = dataset.map(self.parse_single_train)
        elif not train:
            dataset = dataset.map(self.parse_single_test)
        return dataset


if __name__ == '__main__':
    dataloader = DataLoader(100)
    dataset = dataloader.get_dataset('data/input/train.tfrecord')
    print(dataset)
