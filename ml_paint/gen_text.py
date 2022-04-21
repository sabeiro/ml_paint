import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation

import requests
content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
open("data/wonderland.txt", "w", encoding="utf-8").write(content)

sequence_length = 100
BATCH_SIZE = 128
EPOCHS = 30
FILE_PATH = "data/wonderland.txt"
BASENAME = os.path.basename(FILE_PATH)
text = open(FILE_PATH, encoding="utf-8").read()
text = text.lower()
text = text.translate(str.maketrans("", "", punctuation))

n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("unique_chars:", vocab)
n_unique_chars = len(vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)
char2int = {c: i for i, c in enumerate(vocab)}
int2char = {i: c for i, c in enumerate(vocab)}

pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))
encoded_text = np.array([char2int[c] for c in text])
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
for char in char_dataset.take(8):
    print(char.numpy(), int2char[char.numpy()])


sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)
for sequence in sequences.take(2):
    print(''.join([int2char[i] for i in sequence.numpy()]))

def split_sample(sample):
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample)-1) // 2):
        input_ = sample[i: i+sequence_length]
        target = sample[i+sequence_length]
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds

dataset = sequences.flat_map(split_sample)

def one_hot_samples(input_, target):
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

dataset = dataset.map(one_hot_samples)

for element in dataset.take(2):
    print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    print("Target:", int2char[np.argmax(element[1].numpy())])
    print("Input shape:", element[0].shape)
    print("Target shape:", element[1].shape)
    print("="*50, "\n")

ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)


model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

# define the model path
model_weights_path = f"results/{BASENAME}-{sequence_length}.h5"
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
if not os.path.isdir("results"):
    os.mkdir("results")
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)
model.save(model_weights_path)


## generate
import numpy as np
import pickle
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import os
sequence_length = 100
FILE_PATH = "data/wonderland.txt"
BASENAME = os.path.basename(FILE_PATH)
seed = "chapter xiii"
char2int = pickle.load(open(f"{BASENAME}-char2int.pickle", "rb"))
int2char = pickle.load(open(f"{BASENAME}-int2char.pickle", "rb"))
vocab_size = len(char2int)
model.load_weights(f"results/{BASENAME}-{sequence_length}.h5")
s = seed
n_chars = 400
generated = ""
for i in tqdm.tqdm(range(n_chars), "Generating text"):
    X = np.zeros((1, sequence_length, vocab_size))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    predicted = model.predict(X, verbose=0)[0]
    next_index = np.argmax(predicted)
    next_char = int2char[next_index]
    generated += next_char
    seed = seed[1:] + next_char

print("Seed:", s)
print("Generated text:")
print(generated)
