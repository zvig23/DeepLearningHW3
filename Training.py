from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import ModelLSTM
from Preprocessing import prepare_training_data


def get_trained_model(max_lyrics_length,
                      lyrics_sequences,
                      lyrics_vocab_size,
                      preprocessed_melodies,
                      word_index):
    input_lyrics, input_melodies, output_data = prepare_training_data(lyrics_sequences, preprocessed_melodies,
                                                                      lyrics_vocab_size, max_lyrics_length)
    model = ModelLSTM.get_instance(lyrics_input_shape=max_lyrics_length,
                                   melody_feature_size=len(input_melodies[0]),
                                   lyrics_vocab_size=lyrics_vocab_size,
                                   lstm_units=1000,
                                   word_index=word_index,
                                   max_lyrics_length=max_lyrics_length)
    rlop = ReduceLROnPlateau(min_delta=0.01)
    mcp = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)
    model.fit([input_lyrics.squeeze(), input_melodies], output_data, epochs=10, batch_size=64,
              validation_split=0.2, verbose=1, callbacks=[rlop, mcp, tensorboard_callback])

    return model
