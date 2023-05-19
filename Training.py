import ModelLSTM
from Preprocessing import prepare_training_data


def get_trained_model(max_lyrics_length,
                      lyrics_sequences,
                      lyrics_vocab_size,
                      preprocessed_melodies,
                      num_epochs,
                      batch_size, word_index):
    input_lyrics, input_melodies, output_data = prepare_training_data(lyrics_sequences, preprocessed_melodies,
                                                                      lyrics_vocab_size, max_lyrics_length)
    model = ModelLSTM.get_instance(lyrics_input_shape=max_lyrics_length,
                                   melody_feature_size=len(input_melodies[0]),
                                   lyrics_vocab_size=lyrics_vocab_size,
                                   lstm_units=1000,
                                   word_index=word_index)
    model.fit([input_lyrics, input_melodies], output_data, epochs=num_epochs, batch_size=batch_size,
              validation_split=0.2)

    return model
