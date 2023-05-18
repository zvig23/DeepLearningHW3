import ModelLSTM
from Preprocessing import prepare_training_data


def get_trained_model(max_lyrics_length,
                      lyrics_sequences,
                      lyrics_vocab_size,
                      preprocessed_melodies,
                      num_epochs,
                      batch_size):
    input_lyrics, input_melodies, output_data = prepare_training_data(lyrics_sequences, preprocessed_melodies,
                                                                      lyrics_vocab_size, max_lyrics_length)
    model = ModelLSTM.get_instance(lyrics_input_shape=max_lyrics_length,
                                   melody_feature_size=len(input_melodies[0]),
                                   lyrics_vocab_size=lyrics_vocab_size,
                                   embedding_dim=300,
                                   lstm_units=10)
    model.fit([input_lyrics, input_melodies], output_data, epochs=num_epochs, batch_size=batch_size)

    return model
