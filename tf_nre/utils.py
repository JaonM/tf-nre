import tensorflow as tf

text = ["She came here last night"]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text)

word_index = tokenizer.word_index
print(word_index)

test = ["He came last morning"]
seq = tokenizer.texts_to_sequences(test)
print(seq)
