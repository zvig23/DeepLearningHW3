# Load a test melody for generating lyrics
import random

import numpy as np
import pandas as pd
from keras.utils import pad_sequences

from Preprocessing import get_mid_file_path, load_midi_data, lyrics_cleaning, prepare_training_data


def load_test_melody():
    # Load and preprocess the lyrics and melodies data
    lyrics, melodies = [], []
    songs_df = pd.read_csv("../DeepLearningHW3/data/lyrics_test_set.csv", header=None)
    for idx, row in songs_df.iterrows():
        try:
            midi_file_path = get_mid_file_path(row).replace("__", "_")
            midi_data = load_midi_data(midi_file_path)

            clean_lyrics = lyrics_cleaning(row[2])
            lyrics.append(clean_lyrics)

            melodies.append(midi_data)
        except:
            continue
    return lyrics, melodies


# Generate lyrics for a given melody
def generate_lyrics(model, lyrics_sequences, preprocessed_melodies, lyrics_vocab_size, lyrics_tokenizer,
                    max_lyrics_length):
    input_lyrics, input_melodies, output_data = prepare_training_data(lyrics_sequences, preprocessed_melodies,
                                                                      lyrics_vocab_size, max_lyrics_length)
    contexts = np.zeros(len(input_lyrics))

    generated_lyrics = model.predict([input_lyrics.squeeze(), input_melodies])
    generated_lyrics*=lyrics_vocab_size
    for song_id in range(len(input_lyrics)):
        curr_song_context = input_lyrics[song_id]
        contexts[song_id] = input_lyrics[song_id][0][0]
        curr_song_generated_lyrics = generated_lyrics[song_id]
        curr_song_context[0] = np.round(curr_song_generated_lyrics)
    generated_lyrics_ids = lyrics_tokenizer.sequences_to_texts(input_lyrics.squeeze())
    return lyrics_tokenizer.sequences_to_texts([contexts]), generated_lyrics_ids
