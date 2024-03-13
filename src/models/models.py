import tensorflow as tf
from keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Flatten

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

class modelEncoderDecoder(tf.keras.Sequential): 

    #PRE: input_steps, output_steps, input_features and output_features must be a positive int value. The first two must be smaller than the dataset
    #POST: returns a defined and compiled model with a LSTM layer and two dense layers. The optimizer used is adam and the metric is RMSE
    def __init__(self, input_steps, output_steps, input_features, output_features):

        super().__init__()  # Call to basic class constructor
        LEARNING_RATE = 0.000003

        # Add sequential layers to the model
        self.add(LSTM(200, activation='relu', input_shape=(input_steps, input_features)))
        self.add(RepeatVector(output_steps))
        self.add(LSTM(200, activation='relu', return_sequences=True))
        self.add(TimeDistributed(Dense(100, activation='relu')))
        self.add(TimeDistributed(Dense(output_features)))
        self.compile(loss='mse', optimizer='adam')

        # Compile the model
        self.compile(loss=tf.losses.MeanSquaredError(),
                     optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
                     metrics=[tf.metrics.RootMeanSquaredError()])

class modelCnnLstm(tf.keras.Sequential): 
    
    #PRE: input_steps, output_steps, input_features and output_features must be a positive int value. The first two must be smaller than the dataset
    #POST: returns a defined and compiled model with a LSTM layer and two dense layers. The optimizer used is adam and the metric is RMSE
    def __init__(self, input_steps, output_steps, input_features, output_features):

        super().__init__()  # Call to basic class constructor
        LEARNING_RATE = 0.000003

        # Add sequential layers to the model
        self.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_steps, input_features)))
        self.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.add(MaxPooling1D(pool_size=2))
        self.add(Flatten())
        self.add(RepeatVector(output_steps))
        self.add(LSTM(200, activation='relu', return_sequences=True))
        self.add(Dense(output_steps * output_features,'linear'))
        self.add(TimeDistributed(Dense(100, activation='relu')))
        self.add(TimeDistributed(Dense(1)))
        self.compile(loss='mse', optimizer='adam')

        # Compile the model
        self.compile(loss=tf.losses.MeanSquaredError(),
                     optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
                     metrics=[tf.metrics.RootMeanSquaredError()])