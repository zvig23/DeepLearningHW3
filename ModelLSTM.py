from keras import Model
from keras.layers import Input, LSTM, Dense, Embedding, concatenate


def get_instance(lyrics_input_shape,
                 melody_feature_size,
                 lyrics_vocab_size,
                 embedding_dim,
                 lstm_units):  # Define the LSTM model architecture
    lyrics_input = Input(shape=(lyrics_input_shape,))
    melody_input = Input(shape=(melody_feature_size,))
    lyrics_embedding = Embedding(lyrics_vocab_size, embedding_dim)(lyrics_input)
    lstm_output = LSTM(lstm_units)(lyrics_embedding)
    fusion_output = concatenate([lstm_output, melody_input])
    output = Dense(lyrics_vocab_size, activation='softmax')(fusion_output)
    model = Model(inputs=[lyrics_input, melody_input], outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model
