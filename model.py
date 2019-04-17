#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Embedding, Activation, Flatten, Dense, Conv1D, MaxPooling1D, Dropout
from tensorflow.python.keras.models import Model
from data_loader import load_dataset


class Embedder(tf.keras.Model):
    def __init__(self, tokenizer, max_len):
        super(Embedder, self).__init__()
        self.characters_size = len(tokenizer.word_index)
        embedding_weights = self._get_embedding_weights(tokenizer)
        embedding_dim = len(tokenizer.word_index)
        self.embedding_layer = Embedding(self.characters_size + 1,
                                         embedding_dim,
                                         input_length=max_len,
                                         weights=[embedding_weights])

    def call(self, input_tensor):
        x = self.embedding_layer(input_tensor)
        return x

    def _get_embedding_weights(self, tk):
        embedding_weights = []  # (70, 69)
        embedding_weights.append(np.zeros(self.characters_size))  # (0, 69)
        for char, i in tk.word_index.items():  # from index 1 to 69
            onehot = np.zeros(self.characters_size)
            onehot[i - 1] = 1
            embedding_weights.append(onehot)
        embedding_weights = np.array(embedding_weights)
        return embedding_weights

class CharCNN(tf.keras.Model):
    def __init__(self, tokenizer, num_of_classes, dropout_prob=0.5, max_len=1014):
        super(CharCNN, self).__init__()
        self.num_of_classes = num_of_classes
        embedding_dim = 200 

        # Embedding
        self.embedder = Embedder(tokenizer, max_len)

        print(tokenizer.word_index)

        # Convolutional Layer

        #-------Layer 1-------#
        self.conv1 = Conv1D(256, 7, activation='relu')
        self.maxpool1 = MaxPooling1D(pool_size=3)

        #-------Layer 2-------#
        self.conv2 = Conv1D(256, 7, activation='relu')
        self.maxpool2 = MaxPooling1D(pool_size=3)

        #-------Layer 3-------#
        self.conv3 = Conv1D(256, 7, activation='relu')

        #-------Layer 4-------#
        self.conv4 = Conv1D(256, 7, activation='relu')

        #-------Layer 5-------#
        self.conv5 = Conv1D(256, 7, activation='relu')

        #-------Layer 6-------#
        self.conv6 = Conv1D(256, 7, activation='relu')
        self.maxpool3 = MaxPooling1D(pool_size=3)

        self.flatten = Flatten()  # (None, 8704)

        # Fully connected layers

        #-------Layer 7-------#
        self.d1 = Dense(1024, activation='relu')
        self.dropout1 = Dropout(dropout_prob)

        #-------Layer 8-------#
        self.d2 = Dense(1024, activation='relu')
        self.dropout2 = Dropout(dropout_prob)

        #-------Layer 9-------#
        self.d3 = Dense(num_of_classes, activation='softmax')

    def call(self, x):
        x = self.embedder(x)

        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)

        x = self.conv6(x)
        x = self.maxpool3(x)

        x = self.flatten(x) 

        x = self.d1(x)
        x = self.dropout1(x)

        x = self.d2(x)
        x = self.dropout2(x)

        return self.d3(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_of_classes
        return tf.TensorShape(shape)

def create_model(tokenizer, num_of_classes, dropout_prob):
    char_cnn = CharCNN(tokenizer, num_of_classes, dropout_prob)
    inputs = Input(shape=(1014,), name='input')
    outputs = char_cnn(inputs)

    return Model(inputs=inputs, outputs=outputs)

def main():
    train_data, val_data, train_label, val_label, tokenizer = load_dataset() 
    print('train_data[0]', len(train_label[0]))
    char_cnn = CharCNN(tokenizer, len(train_label[0]))

    inputs = Input(shape=(1014,), name='input')
    outputs = char_cnn(inputs)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(train_data, train_label, batch_size=256, epochs=5, validation_data=(val_data,val_label))


#    char_cnn.compile(optimizer=tf.keras.optimizers.Adam(),
#                  loss=tf.keras.losses.categorical_crossentropy,
#                  metrics=['accuracy'])
#
#    char_cnn.fit(train_data, train_label, batch_size=256, epochs=5, validation_data=(val_data,val_label))

if __name__ == "__main__":
    main()

