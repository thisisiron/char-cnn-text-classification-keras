#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from data_loader import load_dataset
from model import CharCNN, create_model 

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("optimizer", "adam", "Select optimizer.")
flags.DEFINE_string("weights_path", "", "")

def test():

    if FLAGS.optimizer=='adam':
        optimizer = tf.keras.optimizers.Adam()
    elif FLAGS.optimizer=='sgd':
        optimizer = tf.keras.optimizers.SGD()


    test_data, test_label, tokenizer = load_dataset('test')

    char_cnn = create_model(tokenizer, len(test_label[0]), 0)

    if os.path.isfile(FLAGS.weights_path): 
        #char_cnn = load_model(FLAGS.weights_path)
        char_cnn.load_weights(FLAGS.weights_path)

    char_cnn.compile(optimizer=optimizer,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    output = char_cnn.evaluate(test_data, test_label)
    print(output)
        

def main():
    test()

if __name__=='__main__':
    main()
