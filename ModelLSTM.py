from keras import Model
from keras.layers import Input, LSTM, Dense, Embedding, concatenate, BatchNormalization, Dropout

import numpy as np


def load_word_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients
    return embeddings_index


def create_embadding_matrix(lyrics_vocab_size, word_index=None):
    word_embeddings = load_word_embeddings('data/wiki-news-300d-1M.vec')
    embedding_dim = 300
    embedding_matrix = np.zeros((lyrics_vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i < lyrics_vocab_size:
            embedding_vector = word_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_instance(lyrics_input_shape,
                 melody_feature_size,
                 lyrics_vocab_size,
                 lstm_units, word_index):  # Define the LSTM model architecture
    lyrics_input = Input(shape=(lyrics_input_shape,))
    melody_input = Input(shape=(melody_feature_size,))

    lyrics_embedding = Embedding(lyrics_vocab_size, 300,
                                 weights=[create_embadding_matrix(lyrics_vocab_size, word_index)])(lyrics_input)
    lstm_output = LSTM(lstm_units)(lyrics_embedding)
    fusion_output = concatenate([lstm_output, melody_input])
    dropout_l = Dropout(0.6)(fusion_output)

    output = Dense(lyrics_vocab_size, activation='softmax')(dropout_l)
    model = Model(inputs=[lyrics_input, melody_input], outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
