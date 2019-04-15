
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Embedding, Activation, Flatten, Dense, Conv1D, MaxPooling1D, Dropout
from tensorflow.python.keras.models import Model

from data_loader import load_dataset


class Embedder(tf.keras.Model):
    def __init__(self, tokenizer, embedding_dim, max_len):
        super(Embedder, self).__init__()
        self.characters_size = len(tokenizer.word_index)
        embedding_weights = self._get_embedding_weights(tokenizer)
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



def model(tokenizer, num_of_classes, dropout_prob=0.5, embedding_dim=69, max_len=1014):

    # Input
    inputs = Input(shape=(max_len,), name='input', dtype='int64')  # shape=(?, 1014)

    # Embedding
    embedder = Embedder(tokenizer, embedding_dim, max_len)
    x = embedder(inputs)

    # Convolutional Layer

    #-------Layer 1-------#
    x = Conv1D(256, 7)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)

    #-------Layer 2-------#
    x = Conv1D(256, 7)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)

    #-------Layer 3-------#
    x = Conv1D(256, 7)(x)
    x = Activation('relu')(x)

    #-------Layer 4-------#
    x = Conv1D(256, 7)(x)
    x = Activation('relu')(x)

    #-------Layer 5-------#
    x = Conv1D(256, 7)(x)
    x = Activation('relu')(x)

    #-------Layer 6-------#
    x = Conv1D(256, 7)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Flatten()(x)  # (None, 8704)

    # Fully connected layers

    #-------Layer 7-------#
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    #-------Layer 8-------#
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    #-------Layer 9-------#
    outputs = Dense(num_of_classes, activation='softmax')(x)

    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model





def main():
    train_data, val_data, train_label, val_label, tokenizer = load_dataset() 
    char_cnn = model(tokenizer, len(train_label[0]))
    char_cnn.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    char_cnn.summary()

if __name__ == "__main__":
    main()

