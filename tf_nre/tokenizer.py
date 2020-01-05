# 简化版 tokenizer


class Tokenizer(object):

    def __init__(self):
        """
        Simple version of tokenizer

        Attributes:
            word_index : Dict stored with token and index
        """
        self.word_index = dict()

    def fit_on_texts(self, texts):
        self.word_index['[PAD]'] = 0
        self.word_index['[UNK]'] = 1
        for text in texts:
            for token in text.split():
                if token not in self.word_index:
                    self.word_index[token] = len(self.word_index)

    def text_to_sequence(self, texts, max_len, padding=True):
        seqs = []
        for text in texts:
            seq = []
            for token in text.split():
                if token in self.word_index:
                    seq.append(self.word_index[token])
                else:
                    seq.append(self.word_index['[UNK]'])
            if len(seq) >= max_len:
                seq = seq[:max_len]
            elif padding:
                seq = seq + [self.word_index['[PAD]']] * (max_len - len(seq))
            seqs.append(seq)
        return seqs
