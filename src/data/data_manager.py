""" 
Runs data processing scripts to turn raw data from (../raw) into
cleaned data ready to be analyzed (saved in ../processed).

TO DO: 
    - Armar documentación de la clase
    - Sumar el codigo que transforma de la data cruda a processed
"""

import pandas as pd
import numpy as np


class DataManager(pd.DataFrame):

    def load_data(self, filename):
        try:
            # Load data into the DataFrame itself
            path = "../data/raw/" + filename
            self.__init__(pd.read_csv(path))
            print(f"Data loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def date_to_timestamp(self): 
        self['date'] = pd.to_datetime(self['date'])
        self['timestamp_date'] = pd.to_datetime(self['date'], format='%m-%d-%Y').map(pd.Timestamp.timestamp)

    def normalize_data(self):
        mean = self.mean()
        std = self.std()
        self[['ppna', 'ppt', 'temp']] = (self[['ppna', 'ppt', 'temp']] - mean[['ppna', 'ppt', 'temp']]) / std[['ppna', 'ppt', 'temp']]

    def denormalize_data(self, mean, std):
        self[['ppna']] = self[['ppna']] * std['ppna'] + mean['ppna']

    """
    This function is main in a LSTM model, prepare the data in form of past observations and future lable. For example, if the data is [1,2,3,4,5,6,7,8,9,10], the seq_len = 5 and the pre_len = 1: 
    past_data = [[1],[2],[3],[4],[5]] , label data = [6]
    past_data = [[2],[3],[4],[5],[6]] , label data = [7]
    and so on...
    """
    def sequence_data_preparation(self, seq_len, pre_len):

        past_data = []  # Window for the past 
        label_data = []  # Predict next value 

        for i in range(self.shape[0] - int(seq_len + pre_len - 1)):
            a = self[i: i + seq_len + pre_len] 
            past_data.append(a[:seq_len])
            label_data.append(a[-pre_len:]['ppna'])

        past_data = np.array(past_data)
        label_data = np.array(label_data)

        return past_data, label_data