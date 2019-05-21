# sequence to vector problem in a sentiment analisis
import tensorflow
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.layers import Dense, Embedding, LSTM

# from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt

print("Loading data..")
# load trainning and test samples
(X_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)

print(X_train[0])
# bacpropagation protection
X_train = sequence.pad_sequeces(x_train, maxlen=80)
x_test = sequence.pad_sequeces(x_test, maxlen=80)

model = Sequential()
# Embedding layer creates a dense vector of fixed size
model.add(Embedding(20000, 128))
# add recurrent NN
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# output layes
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# tranning the model
model.fit(x_train, y_train, batch_size=32, epochs=15, verbose=2, validation_data=(x_test, y_test))


# evaluate the model
score, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=2)

print('Score->', score)
print('accuracy->', acc)