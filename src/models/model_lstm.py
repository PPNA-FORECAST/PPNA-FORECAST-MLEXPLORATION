import tensorflow as tf
from keras.layers import Dense, Input, LSTM

class modelLstm(tf.keras.Sequential): 


    #PRE: input_steps, output_steps, input_features and output_features must be a positive int value. The first two must be smaller than the dataset
    #POST: returns a defined and compiled model with a LSTM layer and two dense layers. The optimizer used is adam and the metric is RMSE
    def __init__(self, input_steps, output_steps, input_features, output_features):
        super().__init__()  # Call to basic class constructor
        LEARNING_RATE = 0.000003

        # Add sequential layers to the model
        self.add(Input((input_steps, input_features)))
        self.add(LSTM(128))
        self.add(Dense(32, activation='relu'))
        self.add(Dense(output_steps * output_features, activation='linear'))

        # Compile the model
        self.compile(loss=tf.losses.MeanSquaredError(),
                     optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
                     metrics=[tf.metrics.RootMeanSquaredError()])
        