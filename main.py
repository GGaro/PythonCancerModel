import pandas as pd

data = pd.read_csv("data/cancer.csv")

x = data.drop(columns=["diagnosis(1=m, 0=b)"])

y = data["diagnosis(1=m, 0=b)"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import tensorflow as tf

model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)

model.evaluate(x_test, y_test)

model.save('cancerModel.h5')