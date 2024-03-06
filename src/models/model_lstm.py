"""
TO DO: 
    - Armar documentación de la clase
"""

import tensorflow as tf
from keras.layers import Dense, Input, LSTM

class modelLstm(tf.keras.Sequential): 

    def __init__(self, input_steps, output_steps, input_features, output_features):
        super().__init__()  # Llama al constructor de la clase base
        LEARNING_RATE = 0.000003

        # Agrega las capas al modelo secuencial
        self.add(Input((input_steps, input_features)))
        self.add(LSTM(128))
        self.add(Dense(32, activation='relu'))
        self.add(Dense(output_steps * output_features, activation='linear'))

        # Compila el modelo
        self.compile(loss=tf.losses.MeanSquaredError(),
                     optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
                     metrics=[tf.metrics.RootMeanSquaredError()])