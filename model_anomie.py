# libraries impoorting
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation

# reading data
df_audio = pd.read_csv('/Users/ivansirokov/Documents/anomie_data/data_fft/spectro_256_ok.csv', sep=';')

# split on test and train
X = df_audio.drop('type', axis=1).astype('float32')
y = df_audio['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100, stratify=y)

# model
model_multiclass = tf.keras.Sequential([
    tf.keras.layers.Dense(555, activation="relu", name="dense_layer1"),
    tf.keras.layers.Dense(539, activation="softmax", name="output_layer")
])

model_multiclass.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                         metrics="accuracy")
model_a = model_multiclass

# fitting the model
model_a.fit(X_train, y_train, epochs=15)

# loading the model to .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model_a)
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the model
with open('/Users/ivansirokov/Documents/anomie_data/data_fft/model_big.tflite', 'wb') as f:
  f.write(tflite_model)