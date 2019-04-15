import tensorflow as tf
from tensorflow import keras

from data_loader import load_dataset
from model import model


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "train or test")

flags.DEFINE_integer("max_len", 1014, "Maximum number of a sequence.")

flags.DEFINE_float("dropout_prob", 0.5, "Probability of dropout.")

flags.DEFINE_bool("data_shuffle", False, "Whether to shuffle dataset.")

flags.DEFINE_bool("do_lower_case", True, "Wheter to lower case the input text.")


def main():
    pass



if __name__=="__main__":
    main()
