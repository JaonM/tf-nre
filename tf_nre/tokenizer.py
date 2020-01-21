# 简化版 tokenizer
import json


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

    def text_to_sequences(self, texts, max_len, padding=True):
        seqs = []
        for text in texts:
            seq = self.text_to_sequence(text, max_len, padding)
            seqs.append(seq)
        return seqs

    def text_to_sequence(self, text, max_len, padding=True):
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
        return seq

    def to_json(self, filename):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.word_index, ensure_ascii=False))

    @staticmethod
    def from_json(filename):
        with open(filename) as f:
            word_index = json.loads(f.readline())
        tokenizer = Tokenizer()
        tokenizer.word_index = word_index
        return tokenizer
