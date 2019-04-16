
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime, timedelta
from data_loader import load_dataset
from model import CharCNN 

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model", "improved", "improved or paper")
flags.DEFINE_string("weights_path", "", "")

def test():
    test_data, test_label, tokenizer = load_dataset(FLAGS.model,
                                                    'test')

    char_cnn = CharCNN(tokenizer, len(test_label[0]), FLAGS.model, 1)

    if os.path.isfile(FLAGS.weights_path): 
        char_cnn.load_weights(FLAGS.weights_path)

    output = char_cnn.evaluate(test_data, test_label)
    print(output)
        

def main():
    test()

if __name__=='__main__':
    main()
