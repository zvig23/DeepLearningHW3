import ModelLSTM
from Preprocessing import prepare_training_data


def get_trained_model(lyrics_sequences=None, preprocessed_melodies=None, num_epochs=None, batch_size=None):
    input_data, output_data = prepare_training_data(lyrics_sequences, preprocessed_melodies)

    model = ModelLSTM.get_instance()

    model.fit([input_data, preprocessed_melodies], output_data, epochs=num_epochs, batch_size=batch_size)

    return model