import numpy as np
from keras.models import Model

import ModelLSTM
import Training
from Evaluation import load_test_melody, generate_lyrics
from Preprocessing import load_data, tokenize_lyrics, preprocess_melodies, prepare_training_data


def main():
    lyrics, melodies = load_data()

    lyrics_tokenizer, lyrics_sequences = tokenize_lyrics(lyrics)
    lyrics_vocab_size = len(lyrics_tokenizer.word_index) + 1
    max_lyrics_length = max(len(sequence) for sequence in lyrics_sequences)

    preprocessed_melodies = preprocess_melodies(melodies)

    model = Training.get_trained_model(max_lyrics_length=max_lyrics_length,
                      lyrics_sequences=lyrics_sequences,
                      lyrics_vocab_size=lyrics_vocab_size,
                      preprocessed_melodies=preprocessed_melodies,
                      num_epochs=10,
                      batch_size=5)

    lyrics, melodies = load_test_melody()
    generated_lyrics = generate_lyrics(model, lyrics, lyrics_tokenizer, max_lyrics_length)

    print(generated_lyrics)

if __name__ == "__main__":
    main()