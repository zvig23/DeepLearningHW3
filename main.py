import Training
from Evaluation import generate_lyrics
from Preprocessing import load_data, tokenize_lyrics, preprocess_melodies


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
                                       num_epochs=1,
                                       batch_size=128)

    # lyrics, melodies = load_test_melody()
    context, generated_lyrics = generate_lyrics(model, lyrics_sequences, preprocessed_melodies, lyrics_vocab_size,
                                                lyrics_tokenizer, max_lyrics_length)

    for i in range(len(context)):
        print("context: " + context[i])
        print("next phrase: "+ generated_lyrics[i])


if __name__ == "__main__":
    main()
