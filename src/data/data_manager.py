""" 
Runs data processing scripts to turn raw data from (../raw) into
cleaned data ready to be analyzed (saved in ../processed).

TO DO: 
    - Sumar el codigo que transforma de la data cruda a processed
    - Modificar normalized_data para que devuelva mean y std de cada columna para desnormalizar
"""

import pandas as pd
import numpy as np


class DataManager(pd.DataFrame):


    #PRE: -
    #POST: loads the .csv info into the DataManager df.
    def load_data(self, filename):
        try:
            # Load data into the DataFrame itself
            path = "../data/raw/" + filename
            self.__init__(pd.read_csv(path))
            print(f"Data loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading data: {e}")


    #PRE: The DataManager df must have a column labeled "date" which contains calendar dates formated as MONTH-DAY-YEAR.
    #POST: A new column labeled "timestamp_date" is added to the df. It contains the date column info presented in a timestamp format.
    def date_to_timestamp(self): 
        self['date'] = pd.to_datetime(self['date'])
        self['timestamp_date'] = pd.to_datetime(self['date'], format='%m-%d-%Y').map(pd.Timestamp.timestamp)


    #PRE: The DataManager df must be previously loaded with "ppna", "temp" and "ppt" columns.
    #POST: "ppna", "temp" and "ppt" columns are now normalised. The function also returns the values of mean and standard deviation of said columns.
    def normalize_data(self):
        mean = self.mean()
        std = self.std()
        self[['ppna', 'ppt', 'temp']] = (self[['ppna', 'ppt', 'temp']] - mean[['ppna', 'ppt', 'temp']]) / std[['ppna', 'ppt', 'temp']]
        return mean, std
        

    #PRE: The DataManager df must be previously normalised and the first column must be "ppna" . 
        # mean and std must be loaded with that information from the normalised columns.
    #POST: de-normalises the "ppna" column from de df.
    def denormalize_data(self, mean, std):
        self[0] = self[0] * std['ppna'] + mean['ppna']


    #PRE: The DataManager df must be previously loaded. seq_len and pre_len must be smaller than the amount of rows in the df.
    #POST: returns two arrays: past_data is an array of arrays, each containing a window of seq_len values for every feature in the df which 
        #  are previous to the row in question and label_data which is an array of arrays, each containing a window of pre_len ppna values in
        #  the df which are all following the row in question
    """
    This function is key in a LSTM model, prepare the data in form of past observations and future lable. For example, if the data is [1,2,3,4,5,6,7,8,9,10], 
    the seq_len = 5 and the pre_len = 1: 
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