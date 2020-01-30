import tensorflow as tf
from tensorflow.keras import layers


class EncoderLayer(layers.Layer):
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
        self.input_len = input_len
        self.num_ex_token = int((k - 1) / 2)

    def call(self, inputs, **kwargs):
        seq_inputs, e1_pos_inputs, e2_pos_inputs = inputs
        seq_inputs, e1_pos_inputs, e2_pos_inputs = self._extra_sequence_padding(seq_inputs), self._extra_pos_padding(
            e1_pos_inputs), self._extra_pos_padding(e2_pos_inputs)
        seq_emb = self.we(seq_inputs)
        e1_pos_emb = self.wpe(e1_pos_inputs)
        e2_pos_emb = self.wpe(e2_pos_inputs)
        seq_tensor = layers.concatenate(
            [seq_emb, e1_pos_emb, e2_pos_emb])  # (batch_size,length+2*num_ex_token,dw+2*dp)

        # generate window tokens to capture contextual features
        tokens = tf.split(seq_tensor, seq_tensor.shape[1], axis=1)
        tokens = list(map(lambda x: tf.squeeze(x, axis=1), tokens))
        contexts = self._capture_contextual_info(tokens)
        contexts = tf.concat(contexts, axis=1)
        return contexts  # (batch_size,length,(dw+2*dp)*k)

    def _capture_contextual_info(self, raw_tokens):
        contexts = []
        for i in range(self.num_ex_token, len(raw_tokens) - self.num_ex_token):
            context = raw_tokens[i - self.num_ex_token:i] + [raw_tokens[i]] + raw_tokens[
                                                                              i + 1:i + self.num_ex_token + 1]
            contexts.append(tf.expand_dims(tf.concat(context, axis=1), axis=1))
        return contexts

    def _extra_sequence_padding(self, tensor):
        """
        extra padding for sequence
        """
        padding = tf.reshape(tf.constant([0]), (-1, 1))
        padding = tf.repeat(padding, self.num_ex_token, axis=1)

        return tf.concat([padding, tensor, padding], axis=1)

    def _extra_pos_padding(self, tensor):
        head_padding_value = tf.reshape(tensor[:, 0], shape=(-1, 1))
        tail_padding_value = tf.reshape(tensor[:, -1], shape=(-1, 1))
        head_padding = tf.repeat(head_padding_value, self.num_ex_token, axis=1)
        tail_padding = tf.repeat(tail_padding_value, self.num_ex_token, axis=1)

        return tf.concat([head_padding, tensor, tail_padding], axis=1)


class EntityAttentionLayer(layers.Layer):
    """
    Compute attention between word and entity

    Parameters:
        dw: Dimension of word embedding
    """

    def __init__(self, dw, input_len, vocab_size, name='subject_att_layer', **kwargs):
        super(EntityAttentionLayer, self).__init__(name, **kwargs)
        self.we = layers.Embedding(vocab_size, dw, input_length=input_len, trainable=True)

    def call(self, inputs, **kwargs):
        entity_seq, token_seq = inputs  # (batch_size,entity_len) (batch_size,seq_len)
        seq_emb = self.we(token_seq)  # (batch_size,seq_len,dw)
        entity_token_emb = self.we(entity_seq)  # tf.RaggedTensor (batch_size,none,dw)
        entity_emb = tf.reduce_mean(entity_token_emb, axis=1)  # (batch_size,dw)

        # evaluate score between entity and tokens
        entity_emb = tf.expand_dims(entity_emb, axis=1)
        seq_emb = tf.transpose(seq_emb, perm=[0, 2, 1])  # (batch_size,dw,seq_len)
        entity_seq = tf.matmul(entity_emb, seq_emb)  # (batch_size,1,seq_len)
        entity_seq = tf.squeeze(entity_seq)
        weight = tf.nn.softmax(entity_seq)
        return weight


if __name__ == '__main__':
    seq, e1_pos, e2_pos = tf.constant([[1, 2, 3, 4]]), tf.constant([[0, 1, 2, 3]]), tf.constant([[4, 5, 6, 7]])
    encoder = EncoderLayer(10, 3, 3, 4, 5)
    print(encoder([seq, e1_pos, e2_pos]).shape)
