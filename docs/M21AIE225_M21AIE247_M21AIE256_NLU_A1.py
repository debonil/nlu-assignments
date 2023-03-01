# %%
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import string
import plotly.graph_objs as go
import plotly.offline as py
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from gensim.models import Word2Vec
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import InputLayer
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# %%
data = pd.read_csv('IMDB Dataset.csv')

print(data)

# %%
sentiments = data['sentiment']
sentiments = sentiments.replace({"positive": 1, "negative": 0})

# %% [markdown]
# 2.Define a text preprocessing pipeline, i.e., stopword removal, lower casing, punctuation removal etc
# 1.Define your own train-val-test split. Ratio : (train: 18: test : 5 , val : 2)

# %%
nltk.download('stopwords')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')
    text = text.replace('\d+', '')

#    text = re.sub(r'\W+', '', text)
    # remove stopwors from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


reviews = data['review'].apply(clean_text)


# %%
X_train, X_test, y_train, y_test = train_test_split(
    reviews, sentiments, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=1)

# %%
print(X_test.shape)
print(X_train.shape)
print(X_val.shape)

# %% [markdown]
# Developing ML methods:
# 1. Model a Naive Bayes classifier.
# a. Count vectorizer features.
# b. TF-IDF features.
#

# %%
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train_count, y_train)

X_test_count = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_count)
accuracy = accuracy_score(y_test, y_pred)


# %%
print(accuracy*100)

# %%
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

X_test_tfidf = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred)

# %%
print(accuracy_tfidf*100)

# %% [markdown]
# 3. ii ) Model a decision tree with TF-IDF features

# %%
clf = DecisionTreeClassifier()
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)
accuracy_dec = accuracy_score(y_test, y_pred)

# %%
print(accuracy_dec*100)

# %% [markdown]
# 4. Developing Deep neural networks:
#
#     a. RNN model.
#
#       1. 64 hidden-vector dimension.
#

# %%
# helps in model building

# %%
# helps in text preprocessing
#from keras.utils import pad_sequences

t = Tokenizer()
t.fit_on_texts(X_train)

# %%
encoded_train = t.texts_to_sequences(X_train)
encoded_test = t.texts_to_sequences(X_test)
encoded_val = t.texts_to_sequences(X_val)
print(encoded_train[0:2])

# %%


# %%
max_length = 128
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
padded_val = pad_sequences(encoded_val, maxlen=max_length, padding='post')
print(padded_train)

# %%
vocab_size = len(t.word_index) + 1
# define the model
RNNModel64 = Sequential(name="SimpleRNNModel64")
RNNModel64.add(Embedding(vocab_size, 64, input_length=max_length))
RNNModel64.add(SimpleRNN(64, return_sequences=False))
RNNModel64.add(Dense(64, activation='relu'))
RNNModel64.add(keras.layers.Dropout(0.5))
RNNModel64.add(Dense(1, activation='relu'))


opt = keras.optimizers.Adam(1e-5)

# compile the model
RNNModel64.compile(optimizer=opt, loss='binary_crossentropy',
                   metrics=['accuracy'])

# summarize the model
print(RNNModel64.summary())

# %%


def plot_training_graph(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# %%
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=5)

# fit the model
history = RNNModel64.fit(x=padded_train,
                         y=y_train,
                         epochs=100,
                         validation_data=(padded_val, y_val), verbose=1,
                         callbacks=[early_stop]
                         )
plot_training_graph(history)

# %%


def c_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy : " + str(acc_sc))
    return acc_sc


def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,
                cmap="Blues", cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# %%
preds_rnn_64 = (RNNModel64.predict(padded_test) > 0.5).astype("int32")

# %%
c_report(y_test, preds_rnn_64)

# %%
plot_confusion_matrix(y_test, preds_rnn_64)

# %% [markdown]
# 4. Developing Deep neural networks:
#
#     a. RNN model.
#
#       2. 256 hidden-vector dimension.
#

# %%
# define the model
RNNModel256 = Sequential(name="SimpleRNNModel256")
RNNModel256.add(Embedding(vocab_size, 256, input_length=max_length))
RNNModel256.add(SimpleRNN(256, return_sequences=False))
RNNModel256.add(Dense(256, activation='relu'))
RNNModel256.add(keras.layers.Dropout(0.5))
RNNModel256.add(Dense(1, activation='relu'))


opt = keras.optimizers.Adam(learning_rate=1e-5)

# compile the model
RNNModel256.compile(
    optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model
print(RNNModel256.summary())

# %%
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=5)

# fit the model
history = RNNModel256.fit(x=padded_train,
                          y=y_train,
                          epochs=100,
                          validation_data=(padded_val, y_val), verbose=1,
                          callbacks=[early_stop]
                          )
plot_training_graph(history)

# %%
preds_rnn_256 = (RNNModel256.predict(padded_test) > 0.5).astype("int32")

# %%
c_report(y_test, preds_rnn_256)

# %%
plot_confusion_matrix(y_test, preds_rnn_256)

# %% [markdown]
# 4. Developing Deep neural networks:
#
#     b. 1-layer LSTM model
#

# %%
# define the model
SingleLSTMLayer64 = Sequential(name="SingleLSTMLayer64")
SingleLSTMLayer64.add(Embedding(vocab_size, 256, input_length=max_length))
SingleLSTMLayer64.add(LSTM(64, return_sequences=False))
SingleLSTMLayer64.add(Dense(64, activation='relu'))
SingleLSTMLayer64.add(keras.layers.Dropout(0.5))
SingleLSTMLayer64.add(Dense(1, activation='relu'))


opt = keras.optimizers.Adam(learning_rate=1e-5)

# compile the model
SingleLSTMLayer64.compile(
    optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model
print(SingleLSTMLayer64.summary())

# %%
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=5)

# fit the model
history = SingleLSTMLayer64.fit(x=padded_train,
                                y=y_train,
                                epochs=100,
                                validation_data=(padded_val, y_val), verbose=1,
                                callbacks=[early_stop]
                                )
plot_training_graph(history)

# %%
preds_1_lstm_64 = (SingleLSTMLayer64.predict(
    padded_test) > 0.5).astype("int32")

# %%
c_report(y_test, preds_1_lstm_64)

# %%
plot_confusion_matrix(y_test, preds_1_lstm_64)

# %% [markdown]
# 4. Developing Deep neural networks:
#
#     c. 2-layer LSTM model
#

# %%
# define the model
DoubleLSTMLayer64 = Sequential(name="DoubleLSTMLayer64")
DoubleLSTMLayer64.add(Embedding(vocab_size, 256, input_length=max_length))
DoubleLSTMLayer64.add(LSTM(64, return_sequences=True))
DoubleLSTMLayer64.add(LSTM(32, return_sequences=False))
DoubleLSTMLayer64.add(Dense(32, activation='relu'))
DoubleLSTMLayer64.add(keras.layers.Dropout(0.5))
DoubleLSTMLayer64.add(Dense(1, activation='relu'))


opt = keras.optimizers.Adam(learning_rate=1e-5)

# compile the model
DoubleLSTMLayer64.compile(
    optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model
print(DoubleLSTMLayer64.summary())

# %%
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=5)

# fit the model
history = DoubleLSTMLayer64.fit(x=padded_train,
                                y=y_train,
                                epochs=100,
                                validation_data=(padded_val, y_val), verbose=1,
                                callbacks=[early_stop]
                                )
plot_training_graph(history)

# %%
preds_2_lstm_64 = (DoubleLSTMLayer64.predict(
    padded_test) > 0.5).astype("int32")

# %%
c_report(y_test, preds_2_lstm_64)

# %%
plot_confusion_matrix(y_test, preds_2_lstm_64)

# %% [markdown]
# 4. Developing Deep neural networks:
#
#     d. 1-layer Bi-LSTM model
#

# %%
# define the model
SingleBiLSTMModel = Sequential(name="SingleBiLSTMModel")
SingleBiLSTMModel.add(Embedding(vocab_size, 256, input_length=max_length))
SingleBiLSTMModel.add(Bidirectional(LSTM(64, return_sequences=False)))
SingleBiLSTMModel.add(Dense(32, activation='relu'))
SingleBiLSTMModel.add(keras.layers.Dropout(0.5))
SingleBiLSTMModel.add(Dense(1, activation='relu'))


opt = keras.optimizers.Adam(learning_rate=1e-5)

# compile the model
SingleBiLSTMModel.compile(
    optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model
print(SingleBiLSTMModel.summary())

# %%
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=5)

# fit the model
history = SingleBiLSTMModel.fit(x=padded_train,
                                y=y_train,
                                epochs=100,
                                validation_data=(padded_val, y_val), verbose=1,
                                callbacks=[early_stop]
                                )
plot_training_graph(history)

# %%
preds_bilstm_64 = (SingleBiLSTMModel.predict(
    padded_test) > 0.5).astype("int32")

# %%
c_report(y_test, preds_bilstm_64)

# %%
plot_confusion_matrix(y_test, preds_bilstm_64)

# %% [markdown]
# 4. Developing Deep neural networks:
#
#     e. Use Google word2vec embeddings as input embedding to model in 4.d. [Compare the performance 4.e vs 4.d]
#

# %%

Embedding_dimensions = 256

# Creating Word2Vec training dataset.
Word2vec_train_data = list(map(lambda x: x.split(), X_train))

# Defining the model and training it.
word2vec_model = Word2Vec(Word2vec_train_data,
                          vector_size=Embedding_dimensions,
                          workers=8,
                          min_count=5)

print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))

# %%
vocab_size = len(t.word_index) + 1
embedding_matrix = np.zeros((vocab_size, Embedding_dimensions))

for word, token in t.word_index.items():
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)

print("Embedding Matrix Shape:", embedding_matrix.shape)

# %%
#max_length = 64

# define the model
BiLSTMLayerWithWord2Vec = Sequential(name="BiLSTMLayerWithWord2Vec")
BiLSTMLayerWithWord2Vec.add(Embedding(vocab_size, 256, input_length=max_length,
                                      weights=[embedding_matrix], trainable=False))
#BiLSTMLayerWithWord2Vec.add(Bidirectional(LSTM(128, dropout=0.3, return_sequences=True)))
BiLSTMLayerWithWord2Vec.add(Bidirectional(
    LSTM(64, dropout=0.3, return_sequences=False)))
BiLSTMLayerWithWord2Vec.add(Dense(32, activation='relu'))
BiLSTMLayerWithWord2Vec.add(Dropout(0.2))
BiLSTMLayerWithWord2Vec.add(Dense(1, activation='relu'))


opt = keras.optimizers.Adam(learning_rate=1e-5)

# compile the model
BiLSTMLayerWithWord2Vec.compile(
    optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model
print(BiLSTMLayerWithWord2Vec.summary())

# %%
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=5)

# fit the model
history = BiLSTMLayerWithWord2Vec.fit(x=padded_train,
                                      y=y_train,
                                      epochs=100,
                                      validation_data=(
                                          padded_val, y_val), verbose=1,
                                      callbacks=[early_stop]
                                      )
plot_training_graph(history)

# %%
preds_2_lstm_64_w2v = (BiLSTMLayerWithWord2Vec.predict(
    padded_test) > 0.5).astype("int32")

# %%
c_report(y_test, preds_2_lstm_64_w2v)
plot_confusion_matrix(y_test, preds_2_lstm_64_w2v)

# %% [markdown]
# Use Glove embeddings as input embedding to model in 4.d. [Compare the performance

# %%
# Keras
# Plotly
py.init_notebook_mode(connected=True)
# Others

# %%
embeddings_index = dict()
#f = open('glove.6B.50d.txt')
embeddings_index = {}
with open('glove.6B.50d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Loaded %s word vectors.' % len(embeddings_index))

# %%
vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(['data'])

sequences = tokenizer.texts_to_sequences(['data'])
data = pad_sequences(sequences, maxlen=50)

# %%
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, 50))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# %%
model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 50, input_length=50,
                weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

# %%
# Tokenize input sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)

# Pad sequences to a fixed length
max_len = 128
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Create input embedding matrix using GloVe embeddings
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define 1-layer Bi-LSTM model with GloVe input embeddings
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[
          embedding_matrix], input_length=max_len, trainable=False))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(padded_sequences, y_train, epochs=10, batch_size=32)

# %%
word_list = []
for word, i in tokenizer.word_index.items():
    word_list.append(word)

# %%


def plot_words(data, start, stop, step):
    trace = go.Scatter(
        x=data[start:stop:step, 0],
        y=data[start:stop:step, 1],
        mode='markers',
        text=word_list[start:stop:step]
    )
    layout = dict(title='glove.50 vs IMDB',
                  yaxis=dict(title='glove.50'),
                  xaxis=dict(title='IMDB'),
                  hovermode='closest')
    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)


# %%
glove_emds = model_glove.layers[0].get_weights()[0]

# %%
glove_tsne_embds = TSNE(n_components=2).fit_transform(glove_emds)

# %%
number_of_words = 2000
plot_words(glove_tsne_embds, 0, number_of_words, 1)

# %%
preds_2_glov_64 = (model.predict(padded_test) > 0.5).astype("int32")

# %%
c_report(y_test, preds_2_glov_64)
plot_confusion_matrix(y_test, preds_2_glov_64)
