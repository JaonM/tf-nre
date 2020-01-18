import re

import tensorflow as tf

from tf_nre.tokenizer import Tokenizer


def parse_text(text):
    """
    process text,return subject&object positions and clean text
    :param text:
    :return: e1 pos,e2pos,clean text
    """
    e1_pos, e2_pos = 0, 0
    tokens = text.split()
    e1_pat = re.compile(r'<e1>.*</e1>')
    e2_pat = re.compile(r'<e2>.*</e2>')
    for i in range(len(tokens)):
        if e1_pat.match(tokens[i]):
            e1_pos = i
            tokens[i] = re.sub('<e1>|</e1>', '', tokens[i])
        if e2_pat.match(tokens[i]):
            e2_pos = i
            tokens[i] = re.sub('<e2>|</e2>', '', tokens[i])
    return e1_pos, e2_pos, ' '.join(tokens)


def read_one_example(lines):
    """
    Return label and raw text
    :return: label,text
    """
    assert len(lines) == 3
    return lines[1], parse_raw_text(lines[0])


def parse_raw_text(line):
    return line.split('\t')[1].strip('"')


def parse_train_example(label_index, tokenizer, label, text, e1_pos, e2_pos, max_len, padding=True):
    """
    Raw text converted to tf.train.Example
    :return:
    """
    seq = tokenizer.text_to_sequence(text, max_len, padding)
    seq_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=seq))
    label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label_index[label]]))
    e1_pos_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[e1_pos]))
    e2_pos_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[e2_pos]))
    feature = {
        'text_seq': seq_feature,
        'e1_pos': e1_pos_feature,
        'e2_pos': e2_pos_feature,
        'label': label_feature
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


def parse_test_example(tokenizer, text, e1_pos, e2_pos, max_len, padding=True):
    """
    Raw text converted to tf.train.Example
    :return:
    """
    seq = tokenizer.text_to_sequence(text, max_len, padding)
    seq_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=seq))
    e1_pos_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[e1_pos]))
    e2_pos_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[e2_pos]))
    feature = {
        'text_seq': seq_feature,
        'e1_pos': e1_pos_feature,
        'e2_pos': e2_pos_feature
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


def read_train(input_path, output_path, max_len, padding=True):
    """
    Read raw train file and write to tfrecord format
    :param input_path:
    :param output_path:
    :return:
    """
    labels, entity_poses, clean_texts = [], [], []
    with open(input_path) as f:
        example_lines = []
        for line in f:
            if not line:
                label, raw_text = read_one_example(example_lines)
                labels.append(label)
                e1_pos, e2_pos, clean_text = parse_text(raw_text)
                entity_poses.append((e1_pos, e2_pos))
                clean_texts.append(clean_text)
                example_lines.clear()
            example_lines.append(line.strip())
    label_index = {}
    for i in range(len(labels)):
        label_index[labels[i]] = i
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_texts)
    with tf.io.TFRecordWriter(output_path) as writer:
        for label, (e1_pos, e2_pos), text in zip(labels, entity_poses, clean_texts):
            example = parse_train_example(label_index, tokenizer, label, text, e1_pos, e2_pos, max_len, padding)
            writer.write(example.SerializeToString())
