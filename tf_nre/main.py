# training process
import json

import numpy as np
import tensorflow as tf

from tf_nre.dataloader import DataLoader
from tf_nre.model import MultiLevelAttCNN
from tf_nre.tokenizer import Tokenizer

# Model Parameters
MAX_LEN = 100
L2_PARAM = 0.001
CONV_SIZE = 100
KERNEL_SIZE = 4
POS_EMB_SIZE = 25
WORD_EMB_SIZE = 300
WINDOW_SIZE = 3
NUM_LABEL = 19

# Training Parameters
NUM_EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.03

TOKENIZER_PATH = 'data/input/tokenizer.json'
TRAIN_DATA_PATH = 'data/input/train.tfrecord'
TEST_DATA_PATH = 'data/input/test.tfrecord'
ID2LABEL_PATH = 'data/input/id2label.json'
RESULT_PATH = 'data/output/prediction.txt'

MODEL_PATH = 'model/'
LABEL_PATH = 'data/input/label2id.json'
EMBED_PATH = 'data/input/glove.840B.300d.txt'


def load_embedding(emb_size):
    word2vec = dict()
    with open(EMBED_PATH) as f:
        for line in f:
            data = line.split()
            word2vec[' '.join(data[0:emb_size])] = np.asarray(data[-emb_size:], np.float32)
    with open(TOKENIZER_PATH) as f:
        word_index = json.loads(f.readline().strip())
    embedding_matrix = np.random.uniform(-1, 1, (len(word_index), WORD_EMB_SIZE))
    for word, index in word_index.items():
        if word in word2vec:
            vector = word2vec[word]
            embedding_matrix[index] = vector
    return embedding_matrix


def compute_label_emb_size(seq_len, kernel_size, padding=0, stride=1):
    size = (seq_len - kernel_size + 2 * padding) / stride + 1
    assert size - int(size) == 0  # 判断size是否为整数
    return int(size)


def loss_fn(predicted, label, label_emb):
    """
    loss function
    :param predicted: tensor (batch_size,num_filter)
    :param label: tensor (batch_size,)
    :param label_emb: tensor (label_size,label_dim)
    :return: loss tensor
    """
    y = locate_label_dim(label, label_emb)
    y_f = locate_farthest_label(predicted, label_emb)
    l = 1 + distance_fn(predicted, y) - distance_fn(predicted, y_f)
    return tf.reduce_mean(l)


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


def train(verbose=False, use_embedding=False):
    dataloader = DataLoader(MAX_LEN)
    dataset = dataloader.get_dataset('data/input/train.tfrecord', True)

    model = init_model(TOKENIZER_PATH, WORD_EMB_SIZE, POS_EMB_SIZE, WINDOW_SIZE, MAX_LEN, CONV_SIZE, KERNEL_SIZE,
                       NUM_LABEL, compute_label_emb_size(MAX_LEN, KERNEL_SIZE), use_embedding)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    dataset = dataset.shuffle(1024*10).batch(BATCH_SIZE)
    dataset = dataset.repeat(NUM_EPOCH)

    for batch_data in dataset:
        text_seq, e1_seq, e2_seq, rel_e1_pos, rel_e2_pos, label = batch_data['text_seq'], batch_data['e1_seq'], \
                                                                  batch_data['e2_seq'], batch_data['rel_e1_pos'], \
                                                                  batch_data['rel_e2_pos'], batch_data['label']
        e1_seq = tf.RaggedTensor.from_sparse(e1_seq)
        e2_seq = tf.RaggedTensor.from_sparse(e2_seq)
        inputs = [tf.cast(text_seq, dtype=tf.int32), tf.cast(rel_e1_pos, dtype=tf.int32),
                  tf.cast(rel_e2_pos, dtype=tf.int32), e1_seq, e2_seq]
        label = tf.squeeze(label)
        train_step(optimizer, model, inputs, label, verbose=verbose)
    # tf.saved_model.save(model, MODEL_PATH)
    model.save_weights(MODEL_PATH)


# @tf.function
def train_step(optimizer, model, inputs, labels, verbose=False):
    with tf.GradientTape() as tape:
        predicted = model(inputs, training=True)
        regularization_loss = tf.math.add_n(model.losses)
        loss = loss_fn(predicted, labels, model.label_emb) + regularization_loss
        if verbose:
            print('current loss', loss.numpy())
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def init_model(tokenizer_filename, word_emb_size, pos_emb_size, window_size, max_len, conv_size, kernel_size, num_label,
               label_emb_size, use_embedding=False):
    tokenizer = Tokenizer.from_json(tokenizer_filename)
    if not use_embedding:
        model = MultiLevelAttCNN(word_emb_size, pos_emb_size, window_size, len(tokenizer.word_index), max_len,
                                 conv_size, kernel_size, num_label, label_emb_size, L2_PARAM)
    else:
        weights = load_embedding(WORD_EMB_SIZE)
        model = MultiLevelAttCNN(word_emb_size, pos_emb_size, window_size, len(tokenizer.word_index), max_len,
                                 conv_size, kernel_size, num_label, label_emb_size, L2_PARAM, weights)
    return model


def test(verbose=False):
    dataloader = DataLoader(MAX_LEN)
    dataset = dataloader.get_dataset('data/input/test.tfrecord', False)
    dataset = dataset.batch(BATCH_SIZE)
    with open(ID2LABEL_PATH) as f:
        id2label = json.loads(f.readline().strip())
    model = init_model(TOKENIZER_PATH, WORD_EMB_SIZE, POS_EMB_SIZE, WINDOW_SIZE, MAX_LEN, CONV_SIZE, KERNEL_SIZE,
                       NUM_LABEL, compute_label_emb_size(MAX_LEN, KERNEL_SIZE), False)
    # model.load_weights(MODEL_PATH)
    labels = []
    if verbose:
        print("start prediction...")
    c_id = 8000
    for batch_data in dataset:
        text_seq, e1_seq, e2_seq, rel_e1_pos, rel_e2_pos = batch_data['text_seq'], batch_data['e1_seq'], \
                                                           batch_data['e2_seq'], batch_data['rel_e1_pos'], \
                                                           batch_data['rel_e2_pos']
        e1_seq = tf.RaggedTensor.from_sparse(e1_seq)
        e2_seq = tf.RaggedTensor.from_sparse(e2_seq)
        inputs = [tf.cast(text_seq, dtype=tf.int32), tf.cast(rel_e1_pos, dtype=tf.int32),
                  tf.cast(rel_e2_pos, dtype=tf.int32), e1_seq, e2_seq]
        preds = model(inputs, training=False).numpy()  # (batch_size,)
        print(preds)
        for pred in preds:
            c_id += 1
            labels.append({"id": c_id, "label": id2label[str(pred)]})
    if verbose:
        print("writing prediction to file...")
        with open(RESULT_PATH, 'w') as f:
            for d in labels:
                f.write(str(d['id']) + '\t' + d['label'])
                f.write('\n')


if __name__ == '__main__':
    # train(True, True)
    test(True)
