# training process
import tensorflow as tf

from tf_nre.dataloader import DataLoader
from tf_nre.model import MultiLevelAttCNN
from tf_nre.tokenizer import Tokenizer

# Model Parameters
MAX_LEN = 100
L2_PARAM = 0.001
CONV_SIZE = 1000
KERNEL_SIZE = 4
POS_EMB_SIZE = 25
WORD_EMB_SIZE = 100
LABEL_EMB_SIZE = 100
WINDOW_SIZE = 3
NUM_LABEL = 19

# Training Parameters
NUM_EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.03

TOKENIZER_PATH = 'data/input/tokenizer.json'
TRAIN_DATA_PATH = 'data/input/train.tfrecord'
TEST_DATA_PATH = 'data/input/test.tfrecord'


def compute_label_emb_size(seq_len, kernel_size, padding=0, stride=1):
    size = (seq_len - kernel_size + 2 * padding) / stride + 1
    assert size - int(size) == 0    # 判断size是否为整数
    return int(size)


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


def train():
    dataloader = DataLoader(MAX_LEN)
    dataset = dataloader.get_dataset('data/input/train.tfrecord', True)

    model = init_model(TOKENIZER_PATH, WORD_EMB_SIZE, POS_EMB_SIZE, WINDOW_SIZE, MAX_LEN, CONV_SIZE, KERNEL_SIZE,
                       NUM_LABEL, LABEL_EMB_SIZE)
    optimizer = tf.keras.optimizers.Adam(0.001)
    for epoch in range(NUM_EPOCH):
        print('Shuffling data...')
        dataset = dataset.shuffle(1024)
        dataset = dataset.batch(BATCH_SIZE)
        for batch_data in dataset:
            text_seq, e1_seq, e2_seq, rel_e1_seq, rel_e2_seq, label = batch_data['text_seq'], batch_data['e1_seq'], \
                                                                      batch_data['e2_seq'], batch_data['rel_e1_seq'], \
                                                                      batch_data['rel_e2_seq'], batch_data['label']
            rel_e1_seq = tf.RaggedTensor.from_sparse(rel_e1_seq)
            rel_e2_seq = tf.RaggedTensor.from_sparse(rel_e2_seq)
            inputs = [text_seq, rel_e1_seq, rel_e2_seq, e1_seq, e2_seq]
            train_step(optimizer, model, inputs, label)
        print('Finishing {} epoch training'.format(epoch))


# @tf.function
def train_step(optimizer, model, inputs, labels):
    with tf.GradientTape() as tape:
        predicted = model(inputs, training=True)
        regularization_loss = tf.math.add_n(model.losses)
        loss = loss_fn(predicted, labels, model.label_emb) + regularization_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def init_model(tokenizer_filename, word_emb_size, pos_emb_size, window_size, max_len, conv_size, kernel_size, num_label,
               label_emb_size):
    tokenizer = Tokenizer.from_json(tokenizer_filename)
    model = MultiLevelAttCNN(word_emb_size, pos_emb_size, window_size, len(tokenizer.word_index), max_len, conv_size,
                             kernel_size, num_label, label_emb_size)
    return model


if __name__ == '__main__':
    train()
