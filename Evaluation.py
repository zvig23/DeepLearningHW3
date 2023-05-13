

# Load a test melody for generating lyrics
import pandas as pd

from Preprocessing import get_mid_file_path, load_midi_data, lyrics_cleaning


def load_test_melody():
    # Load and preprocess the lyrics and melodies data
    lyrics, melodies = [], []
    songs_df = pd.read_csv("data/lyrics_test_set.csv", header=None)
    for idx, row in songs_df.iterrows():
        midi_file_path = get_mid_file_path(row).replace("__","_")
        midi_data = load_midi_data(midi_file_path)

        clean_lyrics = lyrics_cleaning(row[2])
        lyrics.append(clean_lyrics)

        melodies.append(midi_data)

        if idx > 5:
            break
    return lyrics, melodies

# Generate lyrics for a given melody
def generate_lyrics(model, melody, lyrics_tokenizer, max_lyrics_length):
    generated_lyrics=["dvir"]
    return generated_lyrics