import re

import numpy as np
import pretty_midi

import pandas as pd
import mido.midifiles.meta as meta
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

del meta._META_SPECS[0x59]
del meta._META_SPEC_BY_TYPE['key_signature']


def lyrics_cleaning(lyrics):
    clean = lyrics.replace('&', 'eos')
    clean = clean.lower()
    clean = re.sub(' +', ' ', clean)
    return clean


def load_midi_data(midi_file_path):
    return pretty_midi.PrettyMIDI(midi_file_path,)


def load_data():
    # Load and preprocess the lyrics and melodies data
    lyrics, melodies = [], []
    songs_df = pd.read_csv("data/lyrics_train_set.csv", header=None)
    for idx, row in songs_df.iterrows():
        midi_file_path = get_mid_file_path(row)
        midi_data = load_midi_data(midi_file_path)

        clean_lyrics = lyrics_cleaning(row[2])
        lyrics.append(clean_lyrics)

        melodies.append(midi_data)

        if idx > 5:
            break
    return lyrics, melodies


def get_mid_file_path(row):
    artist = row[0].replace(' ', '_')
    song_name = row[1].replace(' ', '_')
    file_name = artist + "_-_" + song_name + ".mid"
    midi_file_path = 'data/midi_files/' + file_name
    return midi_file_path


# Tokenize the lyrics and create word embeddings
def tokenize_lyrics(lyrics):
    lyrics_tokenizer = Tokenizer()
    lyrics_tokenizer.fit_on_texts(lyrics)
    lyrics_sequences = lyrics_tokenizer.texts_to_sequences(lyrics)
    max_lyrics_length = max(len(sequence) for sequence in lyrics_sequences)
    lyrics_sequences = pad_sequences(lyrics_sequences, maxlen=max_lyrics_length, padding='post')
    return lyrics_tokenizer, lyrics_sequences


# Preprocess the melodies
def preprocess_melodies(melodies):
    preprocessed_melodies = []
    for melody_file in melodies:
        preprocessed_melodies.append(preprocess_melody(melody_file))
    return preprocessed_melodies


def preprocess_melody(midi_object=None):
    # Compute the relative amount of each semitone across the entire song, a proxy for key
    total_velocity = sum(sum(midi_object.get_chroma()))
    # Compute the relative amount of each semitone across the entire song, a proxy for key
    semitone = [sum(semitone) / total_velocity for semitone in midi_object.get_chroma()]
    # Compute a piano roll matrix of the MIDI data.
    piano_roll_matrix = midi_object.get_piano_roll()
    piano_roll_norm = piano_roll_matrix.sum(axis=1) / piano_roll_matrix.sum()
    # Computes the frequency of pitch classes of this instrument, optionally weighted by their durations or velocities.
    pc_histogram = midi_object.get_pitch_class_histogram()
    global_tempo_norm = np.array([midi_object.estimate_tempo() / 300])
    melody_data = np.concatenate((semitone, piano_roll_norm, pc_histogram, global_tempo_norm))
    melody_data[np.isnan(melody_data)] = 0
    return melody_data


# Prepare the input-output pairs for training

def prepare_training_data(lyrics_sequences, preprocessed_melodies, lyrics_vocab_size, max_lyrics_length):
    input_lyrics = []
    input_melodies = []
    output_data = []

    for i in range(len(lyrics_sequences) - 1):
        curr_sequence = lyrics_sequences[i]
        for n_gram_size in range(1, curr_sequence.shape[0]):
            context = [curr_sequence[0:n_gram_size]]
            rest_lyrics = [curr_sequence[n_gram_size + 1: len(curr_sequence)]]
            input_sequence = pad_sequences(context, maxlen=len(lyrics_sequences[i]), padding='post')
            output_sequence = pad_sequences(rest_lyrics, maxlen=lyrics_vocab_size, padding='pre')
            input_sequence = input_sequence.squeeze()
            output_sequence = output_sequence.squeeze()
            input_lyrics.append(input_sequence)
            input_melodies.append(preprocessed_melodies[i])
            output_data.append(output_sequence)

    # Convert the input and output data to numpy arrays
    input_lyrics = np.array(input_lyrics)
    input_melodies = np.array(input_melodies)
    output_data = np.array(output_data)

    return input_lyrics, input_melodies, output_data
