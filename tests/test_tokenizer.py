# unit tests
from tf_nre.tokenizer import Tokenizer


def test_fit_on_texts():
    tokenizer = Tokenizer()
    texts = [
        "He is a teacher",
        "I'm a student"
    ]
    word_set = set()
    for t in texts:
        for token in t.split():
            word_set.add(token)
    tokenizer.fit_on_texts(texts)
    assert len(tokenizer.word_index) == len(word_set) + 2


def test_text_to_sequence():
    tokenizer = Tokenizer()
    texts = [
        "He is a teacher",
        "I'm a student"
    ]
    word_set = set()
    for t in texts:
        for token in t.split():
            word_set.add(token)
    tokenizer.fit_on_texts(texts)
    target_texts = ["She is a student too"]
    seqs = tokenizer.text_to_sequences(target_texts, 10)
    print(seqs)
