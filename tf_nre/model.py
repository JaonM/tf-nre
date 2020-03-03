from tf_nre.layers import *


class MultiLevelAttCNN(tf.keras.Model):
    """
    encoder+entity attention+input attention composition+cnn attention pooling+label embedding
    """

    def __init__(self, dw, dp, k, vocab_size, input_len, num_filter, filter_size, label_size, label_dim,
                 name='multi_level_cnn', **kwargs):
        super(MultiLevelAttCNN, self).__init__(name=name, **kwargs)
        # assert num_filter == label_dim
        self.encoder = EncoderLayer(dw, dp, k, input_len, vocab_size)
        self.entity_att = EntityAttentionLayer(dw, input_len, vocab_size)
        self.cnn_att = CNNAttentionLayer(num_filter, filter_size)
        self.label_emb = tf.Variable(tf.random.uniform(shape=[label_size, label_dim], minval=-1, maxval=1))

    def call(self, inputs, training=None, mask=None):
        token_seq, e1_pos_seq, e2_pos_seq, e1_seq, e2_seq = inputs
        contexts = self.encoder([token_seq, e1_pos_seq, e2_pos_seq])  # (batch_size,seq_len,(dw+2dp)*k)
        e1_att = self.entity_att([e1_seq, token_seq])
        e2_att = self.entity_att([e2_seq, token_seq])  # (batch_size,seq_len)
        entity_att = (e1_att + e2_att) / 2
        entity_att = tf.expand_dims(entity_att, axis=2)
        contexts = tf.multiply(entity_att, contexts)
        out = self.cnn_att([contexts, self.label_emb], training=training)  # (batch_size,1) or (batch_size,num_filter)
        return out


