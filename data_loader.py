import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.utils import to_categorical

DECODE = {0: 0, 2: 1, 4: 2} # For One Hot Encdoing


def preprocess(contexts):
    tk = Tokenizer(num_words=None, filters='', char_level=True, oov_token='UNK')
    tk.fit_on_texts(contexts)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
            char_dict[char] = i + 1
            tk.word_index = char_dict.copy()
            tk.word_index[tk.oov_token] = len(char_dict) + 1
    sequence = tk.texts_to_sequences(contexts)
    sequence_pad = pad_sequences(sequence, maxlen=1014, padding='post')
    data = np.array(sequence_pad, dtype='float32')
    print(data)

    return data, tk


def decode_label(label):
    return DECODE[int(label)]

def load_dataset(mode='train', VALID_SPLIT=0.1, shuffle=False):
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    DATA_PATH = './data/'

    if mode=='train':
        data_df = pd.read_csv(DATA_PATH + 'training.1600000.processed.noemoticon.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
        train_data, tk = preprocess(data_df['text'])

        data_df['target'] = data_df['target'].apply(lambda x: decode_label(x))

        label_np = data_df['target'].values
        label_one_hot = to_categorical(label_np)

        print("Dataset size:", len(data_df))

        train_input, val_input, train_label, val_label = train_test_split(train_data, label_one_hot, test_size=VALID_SPLIT, shuffle=shuffle)

        print(train_label)

        return train_input, val_input, train_label, val_label, tk

    elif mode=='test':
        data_df = pd.read_csv(DATA_PATH + '', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

        return



def main():
    load_dataset()

if __name__=="__main__":
    main()
