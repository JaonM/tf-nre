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

    def __init__(self, dw, dp, k, input_len, vocab_size, emb_weights, name='input_encoder', **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        if not emb_weights:
            self.we = layers.Embedding(vocab_size, dw, input_length=input_len, trainable=False)  # word embedding
        else:
            self.we = self.we = layers.Embedding(vocab_size, dw, input_length=input_len,
                                                 embeddings_initializer=tf.keras.initializers.Constant(emb_weights),
                                                 trainable=False)  # pre-trained word embedding
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
        padding = tf.reshape(tf.constant([0] * tensor.shape[0]), (-1, 1))
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

    def __init__(self, dw, input_len, vocab_size, l2, name='subject_att_layer', **kwargs):
        super(EntityAttentionLayer, self).__init__(name, **kwargs)
        self.we = layers.Embedding(vocab_size, dw, input_length=input_len, trainable=True,
                                   embeddings_regularizer=tf.keras.regularizers.l2(l2))

    def call(self, inputs, **kwargs):
        entity_seq, token_seq = inputs  # (batch_size,entity_len) (batch_size,seq_len)
        seq_emb = self.we(token_seq)  # (batch_size,seq_len,dw)
        entity_token_emb = self.we(entity_seq)  # tf.RaggedTensor (batch_size,none,dw)
        entity_emb = tf.reduce_mean(entity_token_emb, axis=1)  # (batch_size,dw)

        # evaluate score between entity and tokens
        entity_emb = tf.expand_dims(entity_emb, axis=1)
        seq_emb = tf.transpose(seq_emb, perm=[0, 2, 1])  # (batch_size,dw,seq_len)
        entity_seq = tf.matmul(entity_emb, seq_emb)  # (batch_size,1,seq_len)
        entity_seq = tf.squeeze(entity_seq,axis=1)
        weight = tf.nn.softmax(entity_seq)
        return weight


class CNNAttentionLayer(layers.Layer):

    def __init__(self, num_filter, filter_size, l2, name='cnn_att_layer', **kwargs):
        super(CNNAttentionLayer, self).__init__(name=name, **kwargs)
        self.cnn = layers.Conv1D(num_filter, filter_size, use_bias=True, activation='tanh',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2),
                                 data_format='channels_last')  # only support channel_last in CPU
        self.num_filter = num_filter

    def build(self, input_shape):
        self.U = self.add_weight(name='weight_matrix', shape=(input_shape[1][-1], input_shape[1][-1]),
                                 initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        """
            inputs: [R,label_emb]
        """
        R, label_emb = inputs  # (batch_size,seq_len,(dw+2dp)*k) (label_size,label_dim)
        R_star = self.cnn(R)  # (batch_size,output_dim,num_filter)
        # R_star = tf.transpose(R_star, perm=[0, 2, 1])   # (batch_size,num_filter,output_dim)
        R_star_T = tf.transpose(R_star, perm=[0, 2, 1])  # (batch_size,num_filter,output_dim)
        label_emb_T = tf.transpose(label_emb)  # (label_dim,label_size)
        G = tf.matmul(R_star_T, self.U)  # (batch_size,num_filter,label_dim)
        G = tf.matmul(G, label_emb_T)  # (batch_size,num_filter,label_size)
        A = tf.nn.softmax(G, axis=1)
        O = tf.matmul(R_star, A)  # (batch_size,output_dim,label_size)
        if 'training' in kwargs and not kwargs['training']:
            O = tf.transpose(O, [0, 2, 1])  # (batch_size,label_size,output_dim)
            O = tf.norm(O - label_emb, axis=2)  # (batch_size,label_size)
            return tf.argmin(O, axis=1)
        else:
            return tf.reduce_max(O, axis=2)  # (batch_size,output_dim)


if __name__ == '__main__':
    seq, e1_pos, e2_pos = tf.constant([[1, 2, 3, 4]]), tf.constant([[0, 1, 2, 3]]), tf.constant([[4, 5, 6, 7]])
    encoder = EncoderLayer(10, 3, 3, 4, 5)
    print(encoder([seq, e1_pos, e2_pos]).shape)
