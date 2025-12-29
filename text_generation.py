import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=1000)
x_train = x_train[:1000]

x = []
y = []

for seq in x_train:
    for i in range(1, len(seq)):
        x.append(seq[:i])
        y.append(seq[i])

x = pad_sequences(x, maxlen=20)
y = np.array(y)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1000, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

model.fit(
    x, y, 
    epochs=3, 
    batch_size=64
)

model.save("text_generation_model.h5")



