# Load a test melody for generating lyrics
import numpy as np
import pandas as pd
from keras.utils import pad_sequences

from Preprocessing import get_mid_file_path, load_midi_data, lyrics_cleaning, prepare_training_data


def load_test_melody():
    # Load and preprocess the lyrics and melodies data
    lyrics, melodies = [], []
    songs_df = pd.read_csv("data/lyrics_train_set.csv", header=None)
    for idx, row in songs_df.iterrows():
        midi_file_path = get_mid_file_path(row).replace("__", "_")
        midi_data = load_midi_data(midi_file_path)

        clean_lyrics = lyrics_cleaning(row[2])
        lyrics.append(clean_lyrics)

        melodies.append(midi_data)

        if idx > 5:
            break
    return lyrics, melodies


# Generate lyrics for a given melody
def generate_lyrics(model, lyrics_sequences, preprocessed_melodies, lyrics_vocab_size, lyrics_tokenizer,max_lyrics_length):
    generated_lyrics_ids = []
    input_lyrics, input_melodies, output_data = prepare_training_data(lyrics_sequences, preprocessed_melodies,
                                                                      lyrics_vocab_size,max_lyrics_length)
    for input_lyric, input_melody  in  zip(input_lyrics, input_melodies):
        lyric_vector = np.squeeze(np.expand_dims(input_lyric, axis=0))
        melody_vector = np.squeeze(np.expand_dims(input_melody, axis=0))
        generated_lyrics = model.predict([lyric_vector, melody_vector])
        for generated_lyric in generated_lyrics:
            next_token_id = np.argmax(generated_lyric)
            generated_lyrics_ids.append(next_token_id)
    contexts = lyrics_tokenizer.sequences_to_texts(input_lyrics)
    generated_lyrics = lyrics_tokenizer.sequences_to_texts(generated_lyrics_ids)
    return contexts,generated_lyrics
