import argparse
import re

import pandas as pd
# Import Keras
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser(description='Language Identification.')
parser.add_argument('--batch_size', type=int,
                    help='Batch Size', default=256)
parser.add_argument('--epochs', type=int, help='Number of Epochs to train on', default=2)
parser.add_argument('--data', type=str, help='Data path', default='all-sentences.txt')
parser.add_argument('--dropout', type=float, help='Dropout level', default=0.5)
parser.add_argument('--embedding_size', type=int, help='embedding size', default=300)

args = parser.parse_args()

batch_size = args.__dict__["batch_size"]
epochs = args.__dict__['epochs']  # Number of epochs to train for.
# Path to the data txt file on disk.
train_data = args.__dict__['data']
dropout = args.__dict__['dropout']
embedding_size = args.__dict__['embedding_size']


def process_sentence(sentence):
    '''Removes all special characters from sentence. It will also strip out
    extra whitespace and makes the string lowercase.
    '''
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|]', '', sentence.lower().strip())


def create_lookup_tables(text):
    """Create lookup tables for vocabulary
    :param text: The text split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab = set(text)

    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {v: k for k, v in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def convert_to_int(data, data_int):
    """Converts all our text to integers
    :param data: The text to be converted
    :return: All sentences in ints
    """
    all_items = []
    for sentence in data:
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])

    return all_items


data = pd.read_csv(train_data, names=["sentence", "language"], header=None, delimiter="|")

# As our sentences in all_sentences.txt are in order, we need to shuffle it first.
sss = StratifiedShuffleSplit(test_size=0.2, random_state=1)

# Clean the sentences
X = data["sentence"].apply(process_sentence)
y = data["language"]

# Split all our sentences
elements = (' '.join([sentence for sentence in X])).split()

X_train, X_test, y_train, y_test = None, None, None, None

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    languages = set(y)
    print("Languages in our dataset: {}".format(languages))
print("Feature Shapes:")
print("\tTrain set: \t\t{}".format(X_train.shape),
      "\n\tTest set: \t\t{}".format(X_test.shape))
print("Totals:\n\tWords in our Dataset: {}\n\tLanguages: {}".format(len(elements), len(languages)))
# Lets look at our training data
print(X_train[:10], y_train[:10])
elements.append("<UNK>")

# Map our vocabulary to int
vocab_to_int, int_to_vocab = create_lookup_tables(elements)
languages_to_int, int_to_languages = create_lookup_tables(y)

print("Vocabulary of our dataset: {}".format(len(vocab_to_int)))

# Convert our inputs# Conver
X_test_encoded = convert_to_int(X_test, vocab_to_int)
X_train_encoded = convert_to_int(X_train, vocab_to_int)

y_data = convert_to_int(y_test, languages_to_int)

encoder = OneHotEncoder()

encoder.fit(y_data)

# One hot encoding our outputs
y_train_encoded = encoder.fit_transform(convert_to_int(y_train, languages_to_int)).toarray()
y_test_encoded = encoder.fit_transform(convert_to_int(y_test, languages_to_int)).toarray()
# Sample of our encoding
print(y_train_encoded[:10], '\n', y_train[:10])
max_sentence_length = 200

# Truncate and pad input sentences
X_train_pad = sequence.pad_sequences(X_train_encoded, maxlen=max_sentence_length)
X_test_pad = sequence.pad_sequences(X_test_encoded, maxlen=max_sentence_length)

# Create the model
model = Sequential()

model.add(Embedding(len(vocab_to_int), embedding_size, input_length=max_sentence_length))
model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
model.add(Bidirectional(LSTM(256, dropout=dropout, recurrent_dropout=dropout)))
model.add(Dense(len(languages), activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# binary_crossentropy
print(model.summary())
# Train the model
model.fit(X_train_pad, y_train_encoded, epochs=epochs, batch_size=batch_size)

# Final evaluation of the model
scores = model.evaluate(X_test_pad, y_test_encoded, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
