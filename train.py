#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime, timedelta
from data_loader import load_dataset
from model import CharCNN, create_model


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("optimizer", "adam", "Select optimizer.")

flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation. (default: 0.1)")

flags.DEFINE_integer("max_len", 1014, "Maximum number of a sequence. (default: 1014)")

flags.DEFINE_float("dropout_prob", 0.5, "Probability of dropout. (default: 0.5)")

flags.DEFINE_bool("data_shuffle", False, "Whether to shuffle dataset. (default: False)")

flags.DEFINE_bool("do_lower_case", True, "Wheter to lower case the input text. (default: True)")

flags.DEFINE_integer("batch_size", 256, "Batch Size. (default: 128)")

flags.DEFINE_integer("num_epochs", 10, "Number of training epochs. (default: 10)")

def save():
    train_begin  = datetime.now()
    MODEL_SAVE_FOLDER_PATH = './weights/' + train_begin.strftime('%m%d%H%M%S') + "/"
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    filepath = MODEL_SAVE_FOLDER_PATH + "{epoch:02d}-{loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    return checkpoint


def main():

    train_data, val_data, train_label, val_label, tokenizer = load_dataset('train',
                                                                           FLAGS.dev_sample_percentage,
                                                                           FLAGS.max_len,) 

    if FLAGS.optimizer=='adam':
        optimizer = tf.keras.optimizers.Adam()
    elif FLAGS.optimizer=='sgd':
        optimizer = tf.keras.optimizers.SGD()

#    char_cnn = CharCNN(tokenizer, len(train_label[0]),  FLAGS.dropout_prob)

    char_cnn = create_model(tokenizer, len(train_label[0]), FLAGS.dropout_prob)
    
    char_cnn.compile(optimizer=optimizer,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    char_cnn.summary()



    callbacks_list = [save()]

    char_cnn.fit(train_data, train_label, batch_size=FLAGS.batch_size, epochs=FLAGS.num_epochs,
                 validation_data=(val_data,val_label), callbacks=callbacks_list)


if __name__=="__main__":
    main()
