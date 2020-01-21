import os
import re

import tensorflow as tf

from tf_nre.tokenizer import Tokenizer

WORD_PUNC_RE = r"(?P<ph>[,./\[\];?!()']*)(?P<w>(<e1>)?(<e2>)?\w+(</e1>)?(</e2>)?('s)?)(?P<pt>[,./\[\];?!()']*)"


def parse_text(text):
    """
    process text,return subject&object positions and clean text
    TODO under modifying
    :param text:
    :return: e1 pos,e2pos,clean text
    """
    e1_pos, e2_pos = [], []
    tokens = text2tokens(text)
    rexp = re.compile(r'<e1>|<e2>|</e1>|</e2>')
    return e1_pos, e2_pos, ' '.join(tokens)


def text2tokens(text):
    """
    preprocess text into tokens
    """
    raw_tokens = text.split()
    rexp = re.compile(WORD_PUNC_RE)
    ret = []
    for token in raw_tokens:
        # split punctuation and word
        tmps = split_token_punctuation(rexp, token)
        for t in tmps:
            if "'s" in t:
                assert t.strip()[-2:] == "'s"
                ret.extend([t.strip()[:-2], "'s"])
            else:
                ret.append(t.strip())
    return ret


def split_token_punctuation(pat, token):
    m = pat.match(token)
    if not m:
        return [token]
    tokens = []
    if m.group('ph'):
        tokens.extend(list(m.group('ph')))
    if m.group('w'):
        tokens.append(m.group('w'))
    if m.group('pt'):
        tokens.extend(list(m.group('pt')))
    return tokens


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
    Read raw train file and convert into tf record format
    :param input_path:
    :param output_path:
    :return:
    """
    labels, entity_poses, clean_texts = [], [], []
    with open(input_path) as f:
        example_lines = []
        for line in f:
            if line == '\n':
                label, raw_text = read_one_example(example_lines)
                labels.append(label)
                e1_pos, e2_pos, clean_text = parse_text(raw_text)
                entity_poses.append((e1_pos, e2_pos))
                clean_texts.append(clean_text)
                example_lines.clear()
            else:
                example_lines.append(line.strip())
    label_index = {}
    labels_unique = list(set(labels))
    for i in range(len(labels_unique)):
        label_index[labels_unique[i]] = i
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_texts)
    # save tokenizer
    filename = os.path.join(os.path.split(output_path)[0], 'tokenizer.json')
    tokenizer.to_json(filename)
    with tf.io.TFRecordWriter(output_path) as writer:
        for label, (e1_pos, e2_pos), text in zip(labels, entity_poses, clean_texts):
            example = parse_train_example(label_index, tokenizer, label, text, e1_pos, e2_pos, max_len, padding)
            writer.write(example.SerializeToString())


def read_test(input_path, output_path, tokenizer, max_len, padding=True):
    """
    Read test file and convert into tf record format
    :return:
    """
    entity_poses, clean_texts = [], []
    with open(input_path, 'r') as f:
        for line in f:
            e1_pos, e2_pos, clean_text = parse_text(parse_raw_text(line))
            entity_poses.append((e1_pos, e2_pos))
            clean_texts.append(clean_text)
    with tf.io.TFRecordWriter(output_path) as writer:
        for (e1_pos, e2_pos), text in zip(entity_poses, clean_texts):
            example = parse_test_example(tokenizer, text, e1_pos, e2_pos, max_len, padding)
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    read_train('data/training/TRAIN_FILE.TXT', 'data/input/train.tfrecord', 100)
    tokenizer = Tokenizer.from_json('data/input/tokenizer.json')
    read_test('data/testing/TEST_FILE.txt', 'data/input/test.tfrecord', tokenizer, 100)
