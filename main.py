import pandas as pd
from keras.utils import pad_sequences

from Evaluation import generate_lyrics, load_test_melody
from Preprocessing import load_data, tokenize_lyrics, preprocess_melodies
from tensorflow import keras

global model, lyrics_vocab_size, max_lyrics_length, tokenizer
import pickle


def main():
    global model, lyrics_vocab_size, max_lyrics_length, tokenizer

    lyrics, melodies = load_data()

    lyrics_tokenizer, lyrics_sequences, tokenizer = tokenize_lyrics(lyrics)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    lyrics_vocab_size = len(lyrics_tokenizer.word_index) + 1
    max_lyrics_length = max(len(sequence) for sequence in lyrics_sequences)

    # preprocessed_melodies = preprocess_melodies(melodies)

    # model = Training.get_trained_model(max_lyrics_length=max_lyrics_length,
    #                                    lyrics_sequences=lyrics_sequences,
    #                                    lyrics_vocab_size=lyrics_vocab_size,
    #                                    preprocessed_melodies=preprocessed_melodies,
    #                                    word_index=lyrics_tokenizer.word_index)

    # lyrics, melodies = load_test_melody()
    # context, generated_lyrics = generate_lyrics(model, lyrics_sequences, preprocessed_melodies, lyrics_vocab_size,
    #                                             lyrics_tokenizer, max_lyrics_length)
    #
    # for i in range(len(context)):
    #     print("context: " + context[i])
    #     print("next phrase: " + generated_lyrics[i])


if __name__ == "__main__":
    # main()
    model = keras.models.load_model('model_checkpoint.h5')
    # loading
    tokenizer = []
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    lyrics, melodies = load_data()
    lyrics_sequences = tokenizer.texts_to_sequences(lyrics)
    lyrics_vocab_size = len(tokenizer.word_index) + 1
    max_lyrics_length = 1484 # max in train
    lyrics_sequences = pad_sequences(lyrics_sequences, maxlen=max_lyrics_length, padding='post').squeeze()
    preprocessed_melodies = preprocess_melodies(melodies)
    context, generated_lyrics = generate_lyrics(model, lyrics_sequences, preprocessed_melodies, lyrics_vocab_size,
                                                tokenizer, max_lyrics_length)
    contexts, generated, originl = [], [], []
    for i in range(len(context)):
        contexts.append(lyrics[i].split(" ")[1])
        generated.append(generated_lyrics[i][0])
        originl.append(" ".join(lyrics[i].split(" ")[1:len(lyrics[i])]))
    results_test = pd.DataFrame(
        {'contexts': contexts,
         'generated': generated,
         'originl': originl
         })
    results_test.to_csv("results/train_results.csv")
