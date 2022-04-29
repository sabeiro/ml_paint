import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from keras.models import model_from_json
from string import punctuation
import string
import tqdm
import re

sequence_length = 100
n_unique_chars = 70

def split_sample(sample):
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample)-1) // 2):
        input_ = sample[i: i+sequence_length]
        target = sample[i+sequence_length]
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds

def one_hot_samples(input_, target):
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

class text_gen():
    def __init__(self,opt):
        self.opt = opt
        if opt['isLoad']: self.load_model()
        else: self.gen_model()
        self.load_coding()

    def clean_text(self,text):
        text = text.lower()
        # text = text.translate(str.maketrans(" "," ", punctuation))
        #text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()
        #text = re.sub("\n"," ",text)
        text = re.sub('"'," ",text)
        text = re.sub("  "," ",text)
        if hasattr(self,'vocab'):
            text = ''.join(c for c in text if c in self.vocab)
        return text

    def gen_coding(self,text):
        text = self.clean_text(text)
        save_path = self.opt['baseDir']+self.opt['cName']
        n_chars = len(text)
        vocab = ''.join(sorted(set(text)))
        print("unique_chars:", vocab)
        n_unique_chars = len(vocab)
        print("Number of characters:", n_chars)
        print("Number of unique characters:", n_unique_chars)
        self.char2int = {c: i for i, c in enumerate(vocab)}
        self.int2char = {i: c for i, c in enumerate(vocab)}
        self.vocab = vocab
        pickle.dump(self.char2int, open(save_path+"-char2int.pickle", "wb"))
        pickle.dump(self.int2char, open(save_path+"-int2char.pickle", "wb"))
        pickle.dump(self.vocab, open(save_path+"-vocab.pickle", "wb"))

    def load_coding(self):
        save_path = self.opt['baseDir']+self.opt['cName']
        try:
            self.char2int = pickle.load(open(save_path+"-char2int.pickle", "rb"))
            self.int2char = pickle.load(open(save_path+"-int2char.pickle", "rb"))
            self.vocab = pickle.load(open(save_path+"-vocab.pickle", "rb"))
        except:
            print('text to int coding not present, please run .gen_coding(text) to generate it')

    def encode_text(self,text):
        text = self.clean_text(text)
        encoded_text = np.array([self.char2int[c] for c in text])
        char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
        sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)
        dataset = sequences.flat_map(split_sample)
        dataset = dataset.map(one_hot_samples)
        ds = dataset.repeat().shuffle(1024).batch(self.opt['batch_size'], drop_remainder=True)
        return ds, len(encoded_text)

    def gen_model(self):
        model = Sequential([
            LSTM(256, input_shape=(sequence_length,n_unique_chars),return_sequences=True),
            Dropout(0.3),
            LSTM(256),
            Dense(n_unique_chars, activation="softmax"),
        ])
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.gen_model = model

    def train(self,n_epoch=200,text='ciccia'):
        if not hasattr(self,'vocab'):
            print('text to int coding not present, please run .gen_coding(text) to generate it')
            return ''
        ds, n_char = self.encode_text(text)
        n_step = (n_char-sequence_length)//self.opt['batch_size']
        self.gen_model.fit(ds,steps_per_epoch=n_step,epochs=n_epoch)
        self.save_model()

    def save_model(self):
        save_path = self.opt['baseDir']+self.opt['cName']
        self.gen_model.save_weights(save_path+"-gen_%s.h5" % (sequence_length))
        model_json = self.gen_model.to_json()
        with open(save_path+"_gen.json", "w") as json_file:
            json_file.write(model_json)
        self.gen_model.save_weights(save_path+"_disc.h5")
        print("model saved")

    def load_model(self):
        save_path = self.opt['baseDir']+self.opt['cName']
        json_file = open(save_path + '_gen.json', 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        json_file.close()
        loaded_model.load_weights(save_path+"-gen_%s.h5" % (sequence_length))
        loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.gen_model = loaded_model
        print('model loaded')

    def gen(self,seed='ciccia',n_chars=sequence_length):
        seed = self.clean_text(seed[:sequence_length])
        s = seed[:]
        generated = ""
        for i in tqdm.tqdm(range(n_chars), "Generating text"):
            X = np.zeros((1, sequence_length, n_unique_chars))
            for t, char in enumerate(seed):
                X[0, (sequence_length - len(seed)) + t, self.char2int[char]] = 1
            predicted = self.gen_model.predict(X, verbose=0)[0]
            next_index = np.argmax(predicted)
            next_char = self.int2char[next_index]
            generated += next_char
            s = s[1:] + next_char

        return generated
