import tensorflow as tf
from tensorflow.keras import layers


class EncoderLayer(tf.keras.layers.Layer):
    """
    Input representation layer

    Parameters:
        dw: Dimension of word embedding
        dp: Dimension of position embedding
        k: Sliding windows size
        input_len: Input sequence length
        vocab_size: Vocabulary size
    """

    def __init__(self, dw, dp, k, input_len, vocab_size, name='input_encoder', **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        self.we = layers.Embedding(vocab_size, dw, input_length=input_len, trainable=False)  # word embedding
        self.wpe = layers.Embedding(2 * input_len, dp, input_length=input_len,
                                    trainable=False)  # word position embedding
        self.window_size = k
        self.num_extra_token = input_len % k

    def call(self, inputs, **kwargs):
        seq_inputs, e1_pos_inputs, e2_pos_inputs = inputs
        seq_inputs, e1_pos_inputs, e2_pos_inputs = self._extra_sequence_padding(seq_inputs), self._extra_pos_padding(
            e1_pos_inputs), self._extra_pos_padding(e2_pos_inputs)
        seq_emb = self.we(seq_inputs)
        e1_pos_emb = self.wpe(e1_pos_inputs)
        e2_pos_emb = self.wpe(e2_pos_inputs)
        seq_concat = layers.concatenate(
            [seq_emb, e1_pos_emb, e2_pos_emb])  # (batch_size,length+2*num_extra_token,dw+2*dp)

        # generate window tokens to capture contextual features

        return seq_concat

    def _extra_sequence_padding(self, tensor):
        """
        extra padding for sequence
        """
        padding = tf.reshape(tf.constant([0]), (-1, 1))
        padding = tf.repeat(padding, self.num_extra_token, axis=1)

        return tf.concat([padding, tensor, padding], axis=1)

    def _extra_pos_padding(self, tensor):
        head_padding_value = tf.reshape(tensor[:, 0], shape=(-1, 1))
        tail_padding_value = tf.reshape(tensor[:, -1], shape=(-1, 1))
        head_padding = tf.repeat(head_padding_value, self.num_extra_token, axis=1)
        tail_padding = tf.repeat(tail_padding_value, self.num_extra_token, axis=1)

        return tf.concat([head_padding, tensor, tail_padding], axis=1)


if __name__ == '__main__':
    seq, e1_pos, e2_pos = tf.constant([[1, 2, 3, 4]]), tf.constant([[0, 1, 2, 3]]), tf.constant([[4, 5, 6, 7]])
    encoder = EncoderLayer(10, 3, 3, 4, 5)
    print(encoder([seq, e1_pos, e2_pos]).shape)
