#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import to_categorical

DECODE = {0: 0, 2: 1, 4: 2} # For One Hot Encdoing
# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9!@#$%_''""\^\&*-.\?]+"
# !@#$%_''""\^\&*-.\?
# @!#\$\^%&*()+=\-\[\]\\\';,\.\/\{\}\|\":<>\?

def preprocess(contexts, max_len, tokenizer=None):

    contexts = contexts.apply(lambda x: re.sub(TEXT_CLEANING_RE, ' ', str(x).lower()).strip())

    if tokenizer == None:
        tk = Tokenizer(filters='', char_level=True, oov_token='UNK')
        tk.fit_on_texts(contexts)
    elif tokenizer != None:
        tk = tokenizer
    
    print(tk.word_index)

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,.:;!?'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
            char_dict[char] = i + 1
            tk.word_index = char_dict.copy()
            tk.word_index[tk.oov_token] = len(char_dict) + 1

    sequence = tk.texts_to_sequences(contexts)
    sequence_pad = pad_sequences(sequence, maxlen=max_len, padding='post')
    data = np.array(sequence_pad, dtype='float32')

    print(data[0:5])
    print(contexts.head())

    return data, tk


def decode_label(label):
    return DECODE[int(label)]

def load_dataset(mode='train', valid_split=0.1, max_len=1014):
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    DATA_PATH = './data/'

    print('Loading...')

    if mode=='train':
        data_df = pd.read_csv(DATA_PATH + 'training.1600000.processed.noemoticon.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
        train_data, tokenizer = preprocess(data_df['text'], max_len)

        data_df['target'] = data_df['target'].apply(lambda x: decode_label(x))

        label_np = data_df['target'].values
        label_one_hot = to_categorical(label_np)

        print("Dataset size:", len(data_df))

        train_input, val_input, train_label, val_label = train_test_split(train_data, label_one_hot, test_size=valid_split, shuffle=True)

        # saving
        with open('tokenizer.pickle', 'wb') as pkle:
            pickle.dump(tokenizer, pkle, protocol=pickle.HIGHEST_PROTOCOL)

        return train_input, val_input, train_label, val_label, tokenizer

    elif mode=='test':
        data_df = pd.read_csv(DATA_PATH + 'testdata.manual.2009.06.14.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

        # loading
        with open('tokenizer.pickle', 'rb') as pkle:
            tokenizer = pickle.load(pkle)

        test_data, tokenizer = preprocess(data_df['text'],  max_len, tokenizer)
        data_df['target'] = data_df['target'].apply(lambda x: decode_label(x))

        label_np = data_df['target'].values
        test_label = to_categorical(label_np)
        
        print("Dataset size:", len(data_df))

        return test_data, test_label, tokenizer



def main():
    load_dataset('test')

if __name__ == "__main__":
    main()
